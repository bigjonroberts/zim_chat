namespace ZimChat

open System.IO
open System.Threading.Tasks

open Microsoft.AspNetCore.Mvc
open Microsoft.AspNetCore.Http

// open Microsoft.Azure.WebJobs
// open Microsoft.Azure.WebJobs.Extensions.Http

open Newtonsoft.Json
open Microsoft.Extensions.Logging

open Microsoft.Azure.Functions.Worker
open Microsoft.Azure.Functions.Worker.Http

// open FsHttp

open OpenAI.Completions.Chat.Api.Cancellable
open OpenAI.Completions.Chat
open System.Net.Http

[<CLIMutable>]
type ChatMessage = {
    Text: string
    Bot: string
}

[<AutoOpen>]
type Chat(logger: ILogger<Chat>, httpClientFactory: IHttpClientFactory) = // httpClient: System.Net.Http.HttpClient) =

    let sendCompletion = Completions.sendAsync Codec.Thoth.asyncRequestCodec<Completion,CompletionResponse>
    let send = Api.Completions.send Codec.Thoth.requestCodec<Completion,CompletionResponse>

    let baseCompletion = Completion.create GPT_3_5_Turbo

    let zimCompletion =
        baseCompletion
        |> Completion.addMessage (Message.create System "You are the zany and irreverant alien Zim, from the hit show Invader Zim.")
        |> function
            | Ok x -> x
            | Error e -> failwith e

    let girCompletion =
        baseCompletion
        |> Completion.addMessage (Message.create System "You are the lovable robot GIR, from the hit show Invader Zim.")
        |> function
            | Ok x -> x
            | Error e -> failwith e

    let buildResponse (req: HttpRequestData) (x: 'a) = task {
        let response = req.CreateResponse(System.Net.HttpStatusCode.OK)
        //response.Headers.Add("Date", "Mon, 18 Jul 2016 16:06:00 GMT")
        response.Headers.Add("Content-Type", "application/json; charset=utf-8")
        let json = Codec.Thoth.Encoder x |> Thoth.Json.Net.Encode.toString 4
        // do! response.WriteAsJsonAsync(x).AsTask()
        do! response.WriteStringAsync(json, System.Text.Encoding.UTF8)
        return response
    }

    [<Function("message")>]
    member _.Run([<HttpTrigger(AuthorizationLevel.Function, "post", Route = "message")>]req: HttpRequestData, executionContext: FunctionContext) = task {
        use logScope = logger.BeginScope("message")
        logger.LogInformation("Chat Message function processed a request.")

        let httpClient = httpClientFactory.CreateClient("openai")

        try
            use stream = new StreamReader(req.Body)
            let! reqBody = stream.ReadToEndAsync()
            
            let message = JsonConvert.DeserializeObject<ChatMessage>(reqBody)
            // let message = Codec.Thoth.Decoder<ChatMessage> reqBody
            // let message = req.ReadFromJsonAsync<ChatMessage>()
            // let query = System.Web.HttpUtility.ParseQueryString(req.QueryString.Value)
            let apiRequest =
                match message.Bot.Trim().ToLower() with
                | "zim" -> Some zimCompletion
                | "gir" -> Some girCompletion
                | _ -> None

            match apiRequest with
            | Some completion -> 
                let apiResponse =
                    completion
                    |> Completion.addNewMessage User message.Text
                    |> fun completion -> logger.LogInformation("Sending completion request: {@completion}", completion); completion
                    |> send (Api.postRequest httpClient)
                //let! apiResponse  =
                //    completion
                //    |> Completion.addNewMessage User message.Text
                //    |> fun completion ->
                //        logger.LogInformation("Sending completion request: {@completion}", completion)
                //        (completion, executionContext.CancellationToken)
                //        |> sendCompletion (postRequestAsync httpClient)
                match apiResponse with
                | Result.Ok r ->
                    printfn "%A" apiResponse
                    return! buildResponse req r
                | Result.Error e ->
                    let response = req.CreateResponse(System.Net.HttpStatusCode.BadRequest)
                    do! response.WriteAsJsonAsync(e, executionContext.CancellationToken).AsTask()
                    return response
            | None ->
                let response = req.CreateResponse(System.Net.HttpStatusCode.BadRequest)
                do! response.WriteAsJsonAsync("Bot must be 'zim' or 'gir'").AsTask()
                return response
                
        with                   
        | ex ->
            logger.LogError(ex, "Error reading request")
            let response = req.CreateResponse(System.Net.HttpStatusCode.InternalServerError)
            //response.Headers.Add("Date", "Mon, 18 Jul 2016 16:06:00 GMT");
            response.Headers.Add("Content-Type", "application/json; charset=utf-8");
            do! response.WriteAsJsonAsync(ex).AsTask()
            return response
    }