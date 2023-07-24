// open Microsoft.Azure.Functions.Extensions.DependencyInjection
open Microsoft.Extensions.DependencyInjection
open Microsoft.Extensions.Hosting
open System.Text.Json
open System.Text.Json.Serialization
// open FsHttp

let apiKey = 
  System.Environment.GetEnvironmentVariable("OPENAI_API_KEY")
  |> Option.ofObj
  |> function
    | Some x -> x
    | None -> failwith "OPENAI_API_KEY not set"

let configureBuilder (builder: Microsoft.Azure.Functions.Worker.IFunctionsWorkerApplicationBuilder) = 
  builder.Services.Configure<JsonSerializerOptions>(fun (options: JsonSerializerOptions) ->
      options.PropertyNamingPolicy <- System.Text.Json.JsonSnakeCaseLowerNamingPolicy() //JorgeSerrano.Json.JsonSnakeCaseNamingPolicy()
      JsonFSharpOptions.Default().AddToJsonSerializerOptions(options)
      // GlobalConfig.Json.defaultJsonSerializerOptions <- options      
      ())
  |> ignore

let host =
  HostBuilder()
    .ConfigureFunctionsWorkerDefaults(configureBuilder)
    .ConfigureServices(fun hostContext services ->
        // see https://learn.microsoft.com/en-us/dotnet/architecture/microservices/implement-resilient-applications/use-httpclientfactory-to-implement-resilient-http-requests
        services.AddHttpClient("openai", fun client -> // if .net 7.0+, we could use Microsoft.Extensions.Options.DefaultName instead of ""
          client.BaseAddress <- System.Uri("https://api.openai.com/v1/")
          client.DefaultRequestHeaders.Authorization <- System.Net.Http.Headers.AuthenticationHeaderValue("Bearer", apiKey)
          ())
          // GlobalConfig.defaults
          // |> Config.setHttpClient client
          // |> GlobalConfig.set)
        |> ignore
      )
    .Build()

host.Run()
