namespace OpenAI.Completions

/// <summary>The OpenAI API is powered by a diverse set of models with different capabilities and price points. You can also make limited customizations to our original base models for your specific use case with fine-tuning.</summary>
/// <remarks>Visit https://platform.openai.com/docs/models/overview for more details</remarks>
type Model =
// <summary>GPT-4 is a large multimodal model (accepting text inputs and emitting text outputs today, with image inputs coming in the future) that can solve difficult problems with greater accuracy than any of our previous models, thanks to its broader general knowledge and advanced reasoning capabilities. Like gpt-3.5-turbo, GPT-4 is optimized for chat but works well for traditional completions tasks both using the Chat Completions API. Learn how to use GPT-4 in our GPT guide.</summary>
/// <summary>More capable than any GPT-3.5 model, able to do more complex tasks, and optimized for chat. Will be updated with our latest model iteration.</summary>
/// <remarks>Max 8,192 tokens
/// Training Data up to Up to Sep 2021</remarks>
| GPT_4
/// <summary>Same capabilities as the base gpt-4 mode but with 4x the context length. Will be updated with our latest model iteration.</summary>
/// <remarks>Max 32,768 tokens
/// Training Data up to Up to Sep 2021</remarks>
| GPT_4_32K
/// <summary>Most capable GPT-3.5 model and optimized for chat at 1/10th the cost of text-davinci-003. Will be updated with our latest model iteration.</summary>
/// <remarks>Max 4,096 tokens
/// Training Data up to Up to Sep 2021</remarks>
| GPT_3_5_Turbo
/// <summary>Can do any language task with better quality, longer output, and consistent instruction-following than the curie, babbage, or ada models. Also supports some additional features such as inserting text.</summary>
/// <remarks>Max 4,097 tokens
/// Training Data up to Up to Jun 2021</remarks>
| Text_Davinci_003
/// <summary>Similar capabilities to text-davinci-003 but trained with supervised fine-tuning instead of reinforcement learning</summary>
/// <remarks>Max 4,097 tokens
/// Training Data up to Up to Jun 2021</remarks>
| Text_Davinci_002
/// <summary>Optimized for code-completion tasks</summary>
/// <remarks>Max 8,001 tokens
/// Training Data up to Up to Jun 2021</remarks>
| Code_Davinci_002
/// Use this constructor to specify a model that is not yet supported by the library
| Other of name: string * contentLength: uint64

module Model =
    let toString =
        function
        | GPT_4 -> "gpt-4"
        | GPT_4_32K -> "gpt-4-32k"
        | GPT_3_5_Turbo -> "gpt-3.5-turbo"
        | Text_Davinci_003 -> "text-davinci-003"
        | Text_Davinci_002 -> "text-davinci-002"
        | Code_Davinci_002 -> "code-davinci-002"
        | Other (s,_) -> s

    let fromString =
        function
        | "gpt-4" -> GPT_4 
        | "gpt-4-32k" -> GPT_4_32K
        | "gpt-3.5-turbo" -> GPT_3_5_Turbo
        | "text-davinci-003" -> Text_Davinci_003
        | "text-davinci-002" -> Text_Davinci_002
        | "code-davinci-002" -> Code_Davinci_002
        | s -> Other (s,1_028UL) //todo: get the max tokens for this model

    let maxTokens =
        function
        | GPT_4 -> 8_192UL
        | GPT_4_32K -> 32_768UL
        | GPT_3_5_Turbo -> 4_096UL
        | Text_Davinci_003 -> 4_097UL
        | Text_Davinci_002 -> 4_097UL
        | Code_Davinci_002 -> 8_001UL
        | Other (_,max_tokens) -> max_tokens

module Chat =

    /// This is a subset of completion models available for chat from https://platform.openai.com/docs/models/model-endpoint-compatibility
    type ChatModel =
    /// <summary>More capable than any GPT-3.5 model, able to do more complex tasks, and optimized for chat. Will be updated with our latest model iteration.</summary>
    /// <remarks>Max 8,192 tokens
    /// Training Data up to Up to Sep 2021</remarks>
    | GPT_4
    /// <summary>Same capabilities as the base gpt-4 mode but with 4x the context length. Will be updated with our latest model iteration.</summary>
    /// <remarks>Max 32,768 tokens
    /// Training Data up to Up to Sep 2021</remarks>
    | GPT_4_32K
    /// <summary>Most capable GPT-3.5 model and optimized for chat at 1/10th the cost of text-davinci-003. Will be updated with our latest model iteration.</summary>
    /// <remarks>Max 4,096 tokens
    /// Training Data up to Up to Sep 2021</remarks>
    | GPT_3_5_Turbo
    /// <summary>Can do any language task with better quality, longer output, and consistent instruction-following than the curie, babbage, or ada models. Also supports some additional features such as inserting text.</summary>
    /// <remarks>Max 4,097 tokens
    /// Training Data up to Up to Jun 2021</remarks>
    /// Use this constructor to specify a model that is not yet supported by the library
    | Other of name: string * contentLength: uint64

    module ChatModel =
        let toModel =
            function
            | GPT_4 -> Model.GPT_4
            | GPT_4_32K -> Model.GPT_4_32K
            | GPT_3_5_Turbo -> Model.GPT_3_5_Turbo
            | Other (n,cl) -> Model.Other (n,cl)
        let fromModel =
            function
            | Model.GPT_4 -> Ok GPT_4
            | Model.GPT_4_32K -> Ok GPT_4_32K
            | Model.GPT_3_5_Turbo -> Ok GPT_3_5_Turbo
            | other -> sprintf "Unsupported chat model: %s" (Model.toString other) |> Error

    type Role =
    | ``System``
    | User
    | Assistant

    module Role =
        let toString =
            function
            | ``System`` -> "system"
            | User -> "user"
            | Assistant -> "assistant"

        let fromString =
            function
            | "system" -> Ok ``System``
            | "user" -> Ok User
            | "assistant" -> Ok Assistant
            | s -> sprintf "Invalid role: %s" s |> Error

    ///Chat message
    type Message = {
        ///The role of the author of this message. One of system, user, or assistant.
        Role: Role
        ///The contents of the message.
        Content: string
        ///The name of the author of this message. May contain a-z, A-Z, 0-9, and underscores, with a maximum length of 64 characters.
        Name: string option //todo: add validation for message author name constraints
    }

    module Message =
        let create (role:Role) (content:string) =
            { Role = role; Content = content; Name = None }

        let private validateName (name: string) =
            let tooLong = name |> String.length > 64
            let invalidChars = name |> String.forall (fun c -> 'a' <= c && c <= 'z' || 'A' <= c && c <= 'Z' || '0' <= c && c <= '9' || c = '_')
            let longMessage = "Message author name cannot be longer than 64 characters"
            let invalidCharsMessage = "Message author name can only contain a-z, A-Z, 0-9, and underscores"
            match (tooLong, invalidChars) with
            | (true, true) -> Error (longMessage + " and " + invalidCharsMessage)
            | (true, false) -> Error longMessage
            | (false, true) -> Error invalidCharsMessage
            | (false, false) -> Ok name

        let createFrom (role:Role) (name:string) (content:string) =
            validateName name
            |> Result.map (fun _ -> { Role = role; Content = content; Name = Some name })

        let addName (name:string) (message:Message) =
            validateName name
            |> Result.map (fun _ -> { message with Name = Some name })

    ///Sampling technique
    type Sampling =
    ///What sampling temperature to use, between 0 and 2. Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic.
    /// - Default: 1
    | Temperature of float
    ///An alternative to sampling with temperature, called nucleus sampling, where the model considers the results of the tokens with top_p probability mass. So 0.1 means only the tokens comprising the top 10% probability mass are considered.
    /// - Default: 1
    | TopP of float
    | Default
    | NotAdvised of temperature:float * topP:float

    ///Chat completion request
    type Completion = {

        ///ID of the model to use. See the model endpoint compatibility table for details on which models work with the Chat API.
        Model: Model

        ///A list of messages describing the conversation so far.
        Messages: Message list
        
        ///What sampling temperature to use, between 0 and 2. Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic.
        ///We generally recommend altering this or top_p but not both.
        Temperature: float option

        ///An alternative to sampling with temperature, called nucleus sampling, where the model considers the results of the tokens with top_p probability mass. So 0.1 means only the tokens comprising the top 10% probability mass are considered.
        ///We generally recommend altering this or temperature but not both.
        TopP: float option

        ///How many completions to generate for each prompt.
        /// - Default: 1
        N: int option

        ///If set, partial message deltas will be sent, like in ChatGPT. Tokens will be sent as data-only server-sent events as they become available, with the stream terminated by a data: [DONE] message. See the OpenAI Cookbook for example code.
        /// - Default: false
        Stream: bool option

        ///Up to 4 sequences where the API will stop generating further tokens.
        Stop: string list option
        
        ///The maximum number of tokens to generate in the chat completion.
        ///The total length of input tokens and generated tokens is limited by the model's context length.
        /// - Default: infinity
        MaxTokens: uint64 option

        ///Number between -2.0 and 2.0. Positive values penalize new tokens based on whether they appear in the text so far, increasing the model's likelihood to talk about new topics.
        /// - Default: 0
        PresencePenalty: float option

        ///Number between -2.0 and 2.0. Positive values penalize new tokens based on their existing frequency in the text so far, decreasing the model's likelihood to repeat the same line verbatim.
        /// - Default: 0
        FrequencePenalty: float option

        ///Modify the likelihood of specified tokens appearing in the completion.
        /// Accepts a json object that maps tokens (specified by their token ID in the tokenizer) to an associated bias value from -100 to 100. Mathematically, the bias is added to the logits generated by the model prior to sampling. The exact effect will vary per model, but values between -1 and 1 should decrease or increase likelihood of selection; values like -100 or 100 should result in a ban or exclusive selection of the relevant token.
        /// - Default: None
        LogitBias: Map<string,sbyte> option

        ///A unique identifier representing your end-user, which can help OpenAI to monitor and detect abuse
        /// - Default: None
        User: string option
    }
        with
            member x.Sampling =
                match (x.Temperature, x.TopP) with
                | (Some t, None) -> Temperature t
                | (None, Some t) -> TopP t
                | (None, None) -> Default
                | (Some t, Some tp) -> NotAdvised (t, tp)

    module Completion =

        let create (model: ChatModel) =
          { Model = model |> ChatModel.toModel
            Messages = List.empty
            Temperature = None
            TopP = None
            N = None
            Stream = None
            Stop = None
            MaxTokens = None
            PresencePenalty = None
            FrequencePenalty = None
            LogitBias = None
            User = None }

        let private checkValue (name: string) (minVal: float, maxVal: float) (value: float) =
            if minVal <= value && value <= maxVal then
                Ok value
            else
                Error (sprintf "%s must be between %f and %f" name minVal maxVal)

        let setSampling (sampling: Sampling) (completion: Completion) =
            match sampling with
            | Temperature t ->
                checkValue "Temperature" (0.0, 2.0) t
                |> Result.map (fun _ -> { completion with Temperature = Some t; TopP = None })
            | TopP tp ->
                checkValue "TopP" (0.0, 1.0) tp
                |> Result.map (fun _ -> { completion with Temperature = None; TopP = Some tp })
            | Default -> Ok { completion with Temperature = None; TopP = None }
            | NotAdvised (t, tp) -> 
                checkValue "Temperature" (0.0, 2.0) t
                |> Result.bind (fun _ -> checkValue "TopP" (0.0, 1.0) tp)
                |> Result.map (fun _ -> { completion with Temperature = Some t; TopP = Some tp })

        let setN (n: int) (completion: Completion) =    
            { completion with N = Some n }
        
        let setStream (stream: bool) (completion: Completion) =    
            { completion with Stream = Some stream }

        let setStop (stop:string list) (completion: Completion) =
            if List.length stop <= 4 then
                Ok { completion with Stop = Some stop }
            else
                Error "Stop list cannot contain more than 4 elements"

        let addStop (stop:string) (completion:Completion) =
            match completion.Stop with
            | None -> Ok { completion with Stop = Some [stop] }
            | Some stops ->
                if 1 + List.length stops <= 4 then
                    Ok { completion with Stop = Some (stop :: stops) }
                else
                    Error "Stop list cannot contain more than 4 elements"

        let removeStop (stop:string) (completion:Completion) =
            match completion.Stop with
            | None -> completion
            | Some stops ->
                { completion with Stop = stops |> List.filter (fun x -> x <> stop) |> Some }

        let clearStop (completion:Completion) =
            { completion with Stop = None }

        let setMaxTokens (maxTokens:uint64) (completion:Completion) =
            { completion with MaxTokens = Some maxTokens }

        let clearMaxTokens (completion:Completion) =
            { completion with MaxTokens = None }

        let setPresencePenalty (presencePenalty:float) (completion:Completion) =
            checkValue "Presence penalty" (-2.0, 2.0) presencePenalty
            |> Result.map (fun _ -> { completion with PresencePenalty = Some presencePenalty })

        let clearPresencePenalty (completion:Completion) =
            { completion with PresencePenalty = None }

        let setFrequencePenalty (frequencePenalty:float) (completion:Completion) =
            checkValue "Frequence penalty" (-2.0, 2.0) frequencePenalty
            |> Result.map (fun _ -> { completion with FrequencePenalty = Some frequencePenalty })

        let clearFrequencePenalty (completion:Completion) =
            { completion with FrequencePenalty = None }

        let setLogitBias (logitBias:Map<string,sbyte>) (completion:Completion) =
            match completion.LogitBias with
            | None -> Ok { completion with LogitBias = Some logitBias }
            | Some biases ->
                if Map.forall (fun _ v -> -100y <= v && v <= 100y) biases then
                    Ok { completion with LogitBias = Some biases }
                else
                    Error "Logit bias values must be between -100 and 100"

        let addLogitBias (token:string) (bias:sbyte) (completion:Completion) =
            match completion.LogitBias with
            | None -> Ok { completion with LogitBias = Map.empty |> Map.add token bias |> Some }
            | Some biases ->
                if -100y <= bias && bias <= 100y then
                    Ok { completion with LogitBias = biases |> Map.add token bias |> Some }
                else
                    Error "Logit bias values must be between -100 and 100"

        let clearLogitBias (completion:Completion) =
            { completion with LogitBias = None }

        let setUser (userId:string) (completion:Completion) =
            { completion with User = Some userId }

        let private checkOneSystemMessage (completion:Completion) =
            completion.Messages
            |> List.filter(fun m -> m.Role = System)
            |> List.length
            |> (=) 1

        let setMessages (messages: Message list) (completion:Completion) =
            if checkOneSystemMessage completion then
                //messages.Sort(fun m1 m2 ->
                //    match (m1.Role, m2.Role) with
                //    | (System, _) -> -1
                //    | (_, System) -> 1
                //    | _ -> 0)
                { completion with
                    Messages = 
                        messages
                        |> List.sortBy(fun m ->
                            match m.Role with
                            | System -> 0
                            | _ -> 1) }
                |> Ok
            else
                Error "One and only one system message must be present in the messages list"

        let addMessage (message:Message) (completion:Completion) =
            match (checkOneSystemMessage completion, message.Role) with
            | (true, System) -> Error "One and only one system message must be present in the messages list"
            | (true, _)
            | (false, System) -> Ok { completion with Messages = message :: completion.Messages }
            | (false, _) -> Error "System message must be the first message in the messages list"

        let addNewMessage (role:Role) (content:string) (completion:Completion) =
            { completion with Messages = completion.Messages @ [ Message.create role content ] }

        let addNewMessagefrom (name:string) (role:Role) (content:string) (completion:Completion) =
            Message.createFrom role name content
            |> Result.map (fun message -> { completion with Messages = completion.Messages @ [ message ] })

    ///Chat completion response
    /// - https://platform.openai.com/docs/api-reference/chat
    [<CLIMutable>]
    type CompletionResponse = {
        ///The ID of the completion.
        Id: string
        ///The object type, which is always "chat.completion".
        Object: string
        ///When the completion was created, in Unix time.
        Created: uint64
        ///A list of possible completions, sorted by the model's confidence score.
        Choices: Choice list
        ///The number of tokens the completion used.
        Usage: Usage option
    }
    and Usage = {
        PromptTokens: uint64
        CompletionTokens: uint64
        TotalTokens: uint64
    }
    and FinishReason =
    | Stop
    | Length
    | Timeout
    | Other of reason: string
    and Choice = {
        ///The index of the choice object in the choices list.
        Index: int
        ///The text of the completion.
        Message: Message
        ///The reason the completion ended. One of: stop, length, or timeout.
        FinishReason: FinishReason
    }

    ///The error message if the request failed.
    /// - https://platform.openai.com/docs/api-reference/errors
    [<CLIMutable>]
    type ErrorResponse = {
        Message: string
        Type: string
        Param: string option
        Code: string option
    }

    type ErrorResponseEnvelope = {
        Error: ErrorResponse
    }

    module FinishReason =
        let toString =
            function
            | Stop -> "stop"
            | Length -> "length"
            | Timeout -> "timeout"
            | Other s -> s
        let fromString =
            function
            | "stop" -> Stop
            | "length" -> Length
            | "timeout" -> Timeout
            | s -> Other s

    module Api =
        open System.Net.Http
        open FsToolkit.ErrorHandling

        let postRequest (client: HttpClient) (path: string) (content: string) =
            use stringContent = new StringContent(content, Headers.MediaTypeHeaderValue("application/json"))
            client.PostAsync(path, stringContent).Result

        let send<'requestObject,'responseObject,'responseError> (codec: Codec.IRequestCodec<'requestObject,'responseObject,'responseError>) (postRequest: string -> HttpResponseMessage) (requestObject: 'requestObject) = 
            requestObject
            |> codec.SerializeRequest
            |> postRequest
            |> (fun response ->
                let stream = response.Content.ReadAsStream()
                use reader = new System.IO.StreamReader(stream)
                reader.ReadToEnd())
            |> codec.DeserializeResponse

        let postRequestAsync (client: HttpClient) (path: string) (contentStream: System.IO.Stream) = task {
            use streamContent = new StreamContent(contentStream)
            streamContent.Headers.ContentType <- Headers.MediaTypeHeaderValue("application/json")
            return! client.PostAsync(path, streamContent)
        }

        let sendAsync<'requestObject,'responseObject,'responseError> (codec: Codec.IAsyncRequestCodec<'requestObject,'responseObject,'responseError>) (postRequestAsync: System.IO.Stream -> System.Threading.Tasks.Task<HttpResponseMessage>) (requestObject: 'requestObject) = task {
            use memoryStream = new System.IO.MemoryStream()
            return!
                (memoryStream,requestObject)
                |> codec.SerializeRequestAsync
                |> Task.bind postRequestAsync
                |> Task.bind (fun response -> response.Content.ReadAsStreamAsync())
                |> Task.bind codec.DeserializeResponseAsync
        }

        module Completions =
            let send (codec: Codec.IRequestCodec<Completion,CompletionResponse,'responseError>) postRequest =
                send codec (postRequest "chat/completions")
            let sendAsync (codec: Codec.IAsyncRequestCodec<Completion,CompletionResponse,'responseError>) postRequestAsync (requestObject, cancellationToken: System.Threading.CancellationToken) =
                sendAsync codec (postRequestAsync "chat/completions")

        module Cancellable =
            let postRequestAsync (client: HttpClient) (path: string) (contentStream: System.IO.Stream, cancellationToken: System.Threading.CancellationToken) = task {                
                use streamContent = new StreamContent(contentStream)
                streamContent.Headers.ContentType <- Headers.MediaTypeHeaderValue("application/json")
                return! client.PostAsync(path, streamContent, cancellationToken)
            }

            let sendAsync<'requestObject,'responseObject>
                (codec: Codec.IAsyncRequestCodec<'requestObject,'responseObject,'responseError>)
                (postRequestAsync: System.IO.Stream * System.Threading.CancellationToken -> System.Threading.Tasks.Task<HttpResponseMessage>)
                (requestObject: 'requestObject, cancellationToken: System.Threading.CancellationToken) = task {
                use memoryStream = new System.IO.MemoryStream()
                return!
                    (memoryStream,requestObject)
                    |> codec.SerializeRequestAsync
                    |> Task.bind (fun stream -> postRequestAsync (stream, cancellationToken))
                    |> Task.bind (fun response -> response.Content.ReadAsStreamAsync())
                    |> Task.bind codec.DeserializeResponseAsync
            }

            module Completions =
                let sendAsync
                    (codec: Codec.IAsyncRequestCodec<Completion,CompletionResponse,'responseError>)
                    postRequestAsync =
                    //(requestObject, cancellationToken: System.Threading.CancellationToken) =
                    sendAsync codec (postRequestAsync "chat/completions")

    module Codec =
        open OpenAI.Completions.Codec

        module SJT =
            open System.Text.Json
            open System.Text.Json.Serialization

            type ModelJsonConverter () =
                inherit JsonConverter<Model>()
                override __.Read (reader: byref<Utf8JsonReader>, typeToConvert: System.Type, options: JsonSerializerOptions) = reader.GetString() |> Model.fromString
                override __.Write( writer: Utf8JsonWriter, modelValue: Model, options: JsonSerializerOptions) =
                    modelValue |> Model.toString |> writer.WriteStringValue

            type RoleJsonConverter () =
                inherit JsonConverter<Role>()
                override __.Read (reader: byref<Utf8JsonReader>, typeToConvert: System.Type, options: JsonSerializerOptions) =
                    match reader.GetString() |> Role.fromString with
                    | Ok role -> role
                    | Error msg -> failwith msg
                override __.Write( writer: Utf8JsonWriter, roleValue: Role, options: JsonSerializerOptions) =
                    roleValue |> Role.toString |> writer.WriteStringValue

            type FinishReasonJsonConverter () =
                inherit JsonConverter<FinishReason>()
                override __.Read (reader: byref<Utf8JsonReader>, typeToConvert: System.Type, options: JsonSerializerOptions) =
                    reader.GetString() |> FinishReason.fromString
                override __.Write( writer: Utf8JsonWriter, finishReasonValue: FinishReason, options: JsonSerializerOptions) =
                    finishReasonValue |> FinishReason.toString |> writer.WriteStringValue


            type private ReaderState =
            | Start
            | ReadingToken of ErrorResponse

            //open System.Reflection

            type ErrorResponseJsonConverter () =
                inherit JsonConverter<ErrorResponse>()

                let snakeCaseLowerNamingPolicy = JsonSnakeCaseLowerNamingPolicy()
                let toSnakeCaseLower (s:string) = snakeCaseLowerNamingPolicy.ConvertName(s)

                let defaultEr = Unchecked.defaultof<ErrorResponse>
                let properties =
                    let t = defaultEr.GetType()
                    t.GetProperties()

                let propLookup =
                  properties
                  |> Array.map (fun prop -> toSnakeCaseLower prop.Name, prop)
                  |> Map.ofArray


                let expectedProperties =
                  //[|nameof defaultEr.Message
                  //  nameof defaultEr.Type
                  //  nameof defaultEr.Param
                  //  nameof defaultEr.Code |]
                  propLookup
                  |> Map.keys
                  |> Set.ofSeq


                override __.Read (reader: byref<Utf8JsonReader>, typeToConvert: System.Type, options: JsonSerializerOptions) =
                    
                    if reader.Read() then
                        if (reader.TokenType <> JsonTokenType.StartObject) then
                            raise (JsonException "Expected StartObject token")
                    else
                        raise (JsonException "Unexpected end of JSON while reading ErrorResponse")

                    // reader passed in by reference leaves us with mutables and loops instead of recursion

                    let mutable errorResponse = {
                        Message = ""
                        Type = ""
                        Param = None
                        Code = None }

                    let mutable foundProperties = Set.empty

                    while reader.Read() do
                        errorResponse <-
                            match reader.TokenType with
                            | JsonTokenType.StartObject ->
                                raise (JsonException "Unexpected StartObject token while reading ErrorResponse")
                            | JsonTokenType.EndObject ->
                                if expectedProperties = foundProperties then
                                    errorResponse
                                else
                                    raise (JsonException "Unexpected end of JSON while reading ErrorResponse")
                            | JsonTokenType.PropertyName ->
                                let propName = reader.GetString()
                                if reader.Read() then
                                    foundProperties <- Set.add propName foundProperties
                                    match Map.tryFind propName propLookup with
                                    | Some property -> property.SetValue(errorResponse, reader.GetString())
                                    | None -> raise (JsonException $"{propName} is not a valid property name for ErrorResponse.")
                                    errorResponse
                                    //match propName with
                                    //| pn when pn = (nameof errorResponse.Message).ToLower() ->
                                    //    { errorResponse with Message = reader.GetString() }
                                    //| pn when pn = (nameof errorResponse.Type).ToLower() ->
                                    //    { errorResponse with Type = reader.GetString() }
                                    //| pn when pn = (nameof errorResponse.Param).ToLower() ->
                                    //    { errorResponse with Param = reader.GetString() |> Some }
                                    //| pn when pn = (nameof errorResponse.Code).ToLower() ->
                                    //    { errorResponse with Code = reader.GetString() |> Some }
                                    //| badPropName -> raise (JsonException $"{badPropName} is not a valid property name for ErrorResponse.")
                                else
                                    raise (JsonException "Unexpected end of JSON while reading ErrorResponse")
                            | _ -> errorResponse
                    errorResponse

                override __.Write (writer: Utf8JsonWriter, errorValue: ErrorResponse, options: JsonSerializerOptions) =
                    JsonSerializer.Serialize(writer, errorValue, options)

            let options =
                let o = JsonSerializerOptions(PropertyNamingPolicy = System.Text.Json.JsonSnakeCaseLowerNamingPolicy())
                [   ModelJsonConverter() :> JsonConverter; RoleJsonConverter(); FinishReasonJsonConverter() ]
                |> List.iter o.Converters.Add
                JsonFSharpOptions
                    .Default()
                    .WithSkippableOptionFields()
                    .WithUnionUnwrapFieldlessTags()
                    .AddToJsonSerializerOptions(o)
                o

            let serialize (x: 'a) =
                JsonSerializer.Serialize(x, options)
                // let json = JsonSerializer.SerializeToUtf8Bytes(x, options)
                // System.Text.Encoding.UTF8.GetString(json)

            let deserialize<'a> (json: string) =
                printf "%s" json
                try
                    JsonSerializer.Deserialize<'a>(json, options) |> Ok
                with
                | :? JsonException as ex ->
                    Error ex.Message
            
            let serializeToStreamAsync (stream: System.IO.Stream) (x: 'a) = task {
                do! JsonSerializer.SerializeAsync<'a>(stream, x, options)
                return stream
            }

            let deserializeAsync<'a> (jsonStream: System.IO.Stream) = task {
                try
                    let! x = JsonSerializer.DeserializeAsync<'a>(jsonStream, options).AsTask()
                    return Ok x
                with
                | :? JsonException as ex ->
                    return Error ex.Message
            }

            module Cancellable =
                let serializeToStreamAsync (stream: System.IO.Stream) (x: 'a, cancellationToken: System.Threading.CancellationToken) = task {
                    do! JsonSerializer.SerializeAsync<'a>(stream, x, options, cancellationToken)
                    return stream
                }

                let deserializeAsync<'a> (jsonStream: System.IO.Stream, cancellationToken: System.Threading.CancellationToken) =  task {
                    try
                        let! x = JsonSerializer.DeserializeAsync<'a>(jsonStream, options, cancellationToken).AsTask()
                        return Ok x
                    with
                    | :? JsonException as ex ->
                        return Error ex.Message
                }

            let requestCodec<'requestObject,'responseObject> = { new IRequestCodec<'requestObject,'responseObject,string> with
                member __.SerializeRequest (requestObject: 'requestObject) = serialize requestObject
                member __.DeserializeResponse (json) = deserialize<'responseObject> json
            }

            let asyncRequestCodec<'requestObject,'responseObject> = { new IAsyncRequestCodec<'requestObject,'responseObject,string> with
                member __.SerializeRequestAsync (stream, requestObject) = serializeToStreamAsync stream requestObject
                member __.SerializeRequestAsync (stream, requestObject, cancellationToken) = Cancellable.serializeToStreamAsync stream (requestObject, cancellationToken)
                member __.DeserializeResponseAsync (jsonStream) = deserializeAsync<'responseObject> jsonStream
                member __.DeserializeResponseAsync (jsonStream, cancellationToken) = Cancellable.deserializeAsync<'responseObject> (jsonStream, cancellationToken)
            }

        module Thoth =
            #if FABLE_COMPILER
            open Thoth.Json
            #else
            open Thoth.Json.Net
            #endif

            module ErrorResponse =
                let encoder : Encoder<ErrorResponse> =
                    fun (x: ErrorResponse) ->
                        Encode.object [
                            "message", Encode.string x.Message
                            "type", Encode.string x.Type
                            "param", Encode.option Encode.string x.Param
                            "code", Encode.option Encode.string x.Code
                        ]

                let decoder : Decoder<ErrorResponse> =
                    Decode.object (fun get -> { 
                        Message = get.Required.Field "message" Decode.string
                        Type = get.Required.Field "type" Decode.string
                        Param = get.Optional.Field "param" Decode.string
                        Code = get.Optional.Field "code" Decode.string })
            
            let resultDecoder<'a> (x: Result<'a,string>) =
                match x with
                | Ok x -> Decode.succeed x
                | Error e -> Decode.fail e

            module ChatModel =
                let encoder : Encoder<ChatModel> = ChatModel.toModel >> Model.toString >> Encode.string
                let decoder : Decoder<ChatModel> =
                    Decode.string
                    |> Decode.andThen (
                        Model.fromString
                        >> ChatModel.fromModel
                        >> resultDecoder)

            module Role =
                let encoder : Encoder<Role> = Role.toString >> Encode.string
                let decoder : Decoder<Role> =
                    Decode.string
                    |> Decode.andThen (
                        Role.fromString
                        >> resultDecoder)

            module Model =
                let encoder: Encoder<Model> = Model.toString >> Encode.string
                let decoder: Decoder<Model> = Decode.string |> Decode.map Model.fromString

            let optionalEncode<'a> (fieldName: string) (encoder: Encoder<'a>) (value: 'a option) =
                value |> Option.map (fun x -> fieldName, encoder x)

            module Message =
                let encoder: Encoder<Message> =
                    fun (x: Message) ->
                      [ "role", x.Role |> Role.toString |> Encode.string
                        "content", x.Content |> Encode.string ]
                      |> List.map Some
                      |> List.append [ x.Name |> optionalEncode "name" Encode.string ] //Option.map (fun name -> "name", Encode.string name) ]
                      |> List.choose id
                      |> Encode.object

                let decoder: Decoder<Message> =
                    Decode.object (fun get -> {
                        Role = get.Required.Field "role" Role.decoder
                        Content = get.Required.Field "content" Decode.string
                        Name = get.Optional.Field "name" Decode.string })

            module Completion =
                let encoder: Encoder<Completion> =
                    fun (x: Completion) ->
                      [ ("model", x.Model |> Model.toString |> Encode.string) |> Some
                        ("messages", x.Messages |> List.map Message.encoder |> Encode.list) |> Some
                        x.Temperature |> optionalEncode "temperature" Encode.float
                        x.TopP |> optionalEncode "top_p" Encode.float
                        x.N |> optionalEncode "n" Encode.int
                        x.Stream |> optionalEncode "stream" Encode.bool
                        x.Stop |> Option.map (List.map Encode.string) |> optionalEncode "stop" Encode.list
                        x.MaxTokens |> optionalEncode "max_tokens" Encode.uint64
                        x.PresencePenalty |> optionalEncode "presence_penalty" Encode.float
                        x.FrequencePenalty |> optionalEncode "frequence_penalty" Encode.float
                        x.LogitBias |> optionalEncode "logit_bias" (Encode.map Encode.string Encode.sbyte)
                        x.User |> optionalEncode "user" Encode.string ]
                      |> List.choose id
                      |> Encode.object
                let decoder: Decoder<Completion> =
                    Decode.object (fun get -> {
                        Model = get.Required.Field "model" Model.decoder
                        Messages = get.Required.Field "messages" (Decode.list Message.decoder)
                        Temperature = get.Optional.Field "temperature" Decode.float
                        TopP = get.Optional.Field "top_p" Decode.float
                        N = get.Optional.Field "n" Decode.int
                        Stream = get.Optional.Field "stream" Decode.bool
                        Stop = get.Optional.Field "stop" (Decode.list Decode.string)
                        MaxTokens = get.Optional.Field "max_tokens" Decode.uint64
                        PresencePenalty = get.Optional.Field "presence_penalty" Decode.float
                        FrequencePenalty = get.Optional.Field "frequence_penalty" Decode.float
                        LogitBias = get.Optional.Field "logit_bias" (Decode.map' Decode.string Decode.sbyte)
                        User = get.Optional.Field "user" Decode.string })

            module Usage =
                let encoder: Encoder<Usage> =
                    fun (x: Usage) -> Encode.object [
                        "prompt_tokens", x.PromptTokens |> Encode.uint64
                        "completion_tokens", x.CompletionTokens |> Encode.uint64
                        "total_tokens", x.TotalTokens |> Encode.uint64 ]
                let decoder: Decoder<Usage> =
                    Decode.object (fun get -> {
                        PromptTokens = get.Required.Field "prompt_tokens" Decode.uint64
                        CompletionTokens = get.Required.Field "completion_tokens" Decode.uint64
                        TotalTokens = get.Required.Field "total_tokens" Decode.uint64 })

            module FinishReason =
                let encoder: Encoder<FinishReason> = FinishReason.toString >> Encode.string
                let decoder: Decoder<FinishReason> = Decode.string |> Decode.map FinishReason.fromString

            module Choice =
                let encoder: Encoder<Choice> =
                    fun (x: Choice) -> Encode.object [
                        "index", x.Index |> Encode.int
                        "message", x.Message |> Message.encoder
                        "finish_reason", x.FinishReason |> FinishReason.encoder ]
                let decoder: Decoder<Choice> =
                    Decode.object (fun get -> {
                        Index = get.Required.Field "index" Decode.int
                        Message = get.Required.Field "message" Message.decoder
                        FinishReason = get.Required.Field "finish_reason" FinishReason.decoder })

            module CompletionResponse =
                let encoder: Encoder<CompletionResponse> =
                    fun (x: CompletionResponse) -> Encode.object [
                        "id", x.Id |> Encode.string
                        "object", x.Object |> Encode.string
                        "created", x.Created |> Encode.uint64
                        "choices", x.Choices |> List.map Choice.encoder |> Encode.list
                        "usage", x.Usage |> Encode.option Usage.encoder ]

                let decoder: Decoder<CompletionResponse> =
                    Decode.object (fun get -> {
                        Id = get.Required.Field "id" Decode.string
                        Object = get.Required.Field "object" Decode.string
                        Created = get.Required.Field "created" Decode.uint64
                        Choices = get.Required.Field "choices" (Decode.list Choice.decoder)
                        Usage = get.Optional.Field "usage" Usage.decoder })

            let codecs =
                Extra.empty
                |> Extra.withUInt64
                |> Extra.withCustom ErrorResponse.encoder ErrorResponse.decoder
                |> Extra.withCustom Model.encoder Model.decoder
                |> Extra.withCustom ChatModel.encoder ChatModel.decoder
                |> Extra.withCustom Role.encoder Role.decoder
                |> Extra.withCustom Message.encoder Message.decoder
                |> Extra.withCustom Completion.encoder Completion.decoder
                |> Extra.withCustom Usage.encoder Usage.decoder
                |> Extra.withCustom FinishReason.encoder FinishReason.decoder
                |> Extra.withCustom Choice.encoder Choice.decoder
                |> Extra.withCustom CompletionResponse.encoder CompletionResponse.decoder

            let inline Encoder<'T> = Encode.Auto.generateEncoderCached<'T>(caseStrategy = SnakeCase, extra = codecs)
            let inline Decoder<'T> = Decode.Auto.generateDecoderCached<'T>(caseStrategy = SnakeCase, extra = codecs)

            let serialize (x: 'a) =
                let json = Encoder x |> Encode.toString 4
                printfn "%s" json
                json

            module Result =
                let decoder<'a> : Decoder<Result<'a,ErrorResponse>> =
                    [   Decoder<'a> |> Decode.map Ok
                        Decoder<ErrorResponseEnvelope> |> Decode.map (fun x -> Error x.Error) ]
                    |> Decode.oneOf

                    //let decodeErrorResponse =
                    //    Decode.object (fun get ->
                    //        let messageField = get.Optional.Field "message" Decode.string
                    //        let typeField = get.Optional.Field "type" Decode.string
                    //        match (messageField, typeField) with
                    //        | (Some message, Some type') ->
                    //            {   Message = message
                    //                Type = type'
                    //                Param = get.Optional.Field "param" Decode.string
                    //                Code = get.Optional.Field "code" Decode.string }
                    //            |> Error
                    //            |> Ok
                    //        | _ -> Error "not an error response"
                    //    )

            type ResponseError =
            | ErrorResponse of ErrorResponse
            | JsonError of string

            let deserialize<'a> (json: string) =
                //printf "%s" json
                match Decode.fromString Result.decoder<'a> json with
                | Ok x -> x |> Result.mapError ErrorResponse
                | Error e -> e |> JsonError |> Error
            
            let serializeToStreamAsync<'a> (stream: System.IO.Stream) (x: 'a) = task {
                do!
                    x
                    |> serialize
                    |> System.Text.Encoding.UTF8.GetBytes
                    |> fun b -> stream.WriteAsync(b).AsTask()
                return stream
            }

            let deserializeAsync<'a> (jsonStream: System.IO.Stream) = task {
                use reader = new System.IO.StreamReader(jsonStream, System.Text.Encoding.UTF8)
                let! json = reader.ReadToEndAsync()
                return deserialize<'a> json
            }

            module private CancellableInternal =
                let serializeToStreamAsync<'a> (stream: System.IO.Stream) (x: 'a, cancellationToken: System.Threading.CancellationToken) = task {
                    do!
                        x
                        |> serialize
                        |> System.Text.Encoding.UTF8.GetBytes
                        |> fun b -> stream.WriteAsync(b, cancellationToken).AsTask()
                    return stream
                }

                let deserializeAsync<'a> (jsonStream: System.IO.Stream, cancellationToken: System.Threading.CancellationToken) = task {
                    use reader = new System.IO.StreamReader(jsonStream, System.Text.Encoding.UTF8)
                    let! json = reader.ReadToEndAsync cancellationToken
                    return deserialize<'a> json
                }

            let requestCodec<'requestObject,'responseObject> = { new IRequestCodec<'requestObject,'responseObject,ResponseError> with
                member __.SerializeRequest (requestObject: 'requestObject) = serialize requestObject
                member __.DeserializeResponse (json) = deserialize<'responseObject> json
            }

            let asyncRequestCodec<'requestObject,'responseObject> = { new IAsyncRequestCodec<'requestObject,'responseObject,ResponseError> with
                member __.SerializeRequestAsync (stream, requestObject) = serializeToStreamAsync<'requestObject> stream requestObject
                member __.SerializeRequestAsync (stream, requestObject, cancellationToken) = CancellableInternal.serializeToStreamAsync<'requestObject> stream (requestObject, cancellationToken)
                member __.DeserializeResponseAsync (jsonStream) = deserializeAsync<'responseObject> jsonStream
                member __.DeserializeResponseAsync (jsonStream, cancellationToken) = CancellableInternal.deserializeAsync<'responseObject> (jsonStream, cancellationToken)
            }

            module Completions =
                let send = Api.send requestCodec<Completion,CompletionResponse>
                let sendAsync = Api.sendAsync asyncRequestCodec<Completion,CompletionResponse>

            module Cancellable =
                module Completions =
                   let sendAsync = Api.Cancellable.sendAsync asyncRequestCodec<Completion,CompletionResponse>
