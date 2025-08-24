use std::net::SocketAddr;
use warp::Filter;

#[tokio::main]
async fn main() {
    // Create a mock response for the chat completions endpoint
    let mock_response = warp::post()
        .and(warp::path("v1"))
        .and(warp::path("chat"))
        .and(warp::path("completions"))
        .map(|| {
            warp::reply::json(&serde_json::json!({
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": "This is a test summary of the example document.\
            \
            - Point 1\
            - Point 2\
            - Point 3"
                    }
                }]
            }))
        });

    // Start the server
    let server = warp::serve(mock_response);
    let addr: SocketAddr = ([127, 0, 0, 1], 3030).into();
    println!("Mock OpenAI API server running on http://{}", addr);

    server.run(addr).await;
}
