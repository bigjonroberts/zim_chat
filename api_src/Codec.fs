namespace OpenAI.Completions.Codec

open System.Threading.Tasks

//type IRequestCodec<'requestObject,'responseObject> =
//    abstract member SerializeRequest: 'requestObject -> string
//    abstract member DeserializeResponse: string -> Result<'responseObject,string>

//type IAsyncRequestCodec<'requestObject,'responseObject> =
//    abstract member SerializeRequestAsync: System.IO.Stream * 'requestObject -> Task<System.IO.Stream>
//    abstract member SerializeRequestAsync: System.IO.Stream * 'requestObject * System.Threading.CancellationToken -> Task<System.IO.Stream>
//    abstract member DeserializeResponseAsync: System.IO.Stream -> Task<Result<'responseObject,string>>
//    abstract member DeserializeResponseAsync: System.IO.Stream * System.Threading.CancellationToken -> Task<Result<'responseObject,string>>

//type IRequestCodec<'requestObject,'responseObject> =
//    abstract member SerializeRequest: 'requestObject -> string
//    abstract member DeserializeResponse: string -> 'responseObject

//type IAsyncRequestCodec<'requestObject,'responseObject> =
//    abstract member SerializeRequestAsync: System.IO.Stream * 'requestObject -> Task<System.IO.Stream>
//    abstract member SerializeRequestAsync: System.IO.Stream * 'requestObject * System.Threading.CancellationToken -> Task<System.IO.Stream>
//    abstract member DeserializeResponseAsync: System.IO.Stream -> Task<'responseObject>
//    abstract member DeserializeResponseAsync: System.IO.Stream * System.Threading.CancellationToken -> Task<'responseObject>

type IRequestCodec<'requestObject,'responseObject,'responseError> =
    abstract member SerializeRequest: 'requestObject -> string
    abstract member DeserializeResponse: string -> Result<'responseObject,'responseError>

type IAsyncRequestCodec<'requestObject,'responseObject,'responseError> =
    abstract member SerializeRequestAsync: System.IO.Stream * 'requestObject -> Task<System.IO.Stream>
    abstract member SerializeRequestAsync: System.IO.Stream * 'requestObject * System.Threading.CancellationToken -> Task<System.IO.Stream>
    abstract member DeserializeResponseAsync: System.IO.Stream -> Task<Result<'responseObject,'responseError>>
    abstract member DeserializeResponseAsync: System.IO.Stream * System.Threading.CancellationToken -> Task<Result<'responseObject,'responseError>>

