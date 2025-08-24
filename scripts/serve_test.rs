use axum::{
    routing::get,
    Router,
};
use std::net::SocketAddr;
use tokio::fs;

async fn serve_file() -> Result<String, (axum::http::StatusCode, String)> {
    match fs::read_to_string("fixtures/html/test.html").await {
        Ok(content) => Ok(content),
        Err(e) => Err((axum::http::StatusCode::INTERNAL_SERVER_ERROR, format!("Failed to read file: {}", e))),
    }
}

#[tokio::main]
async fn main() {
    // Build our application with a single route
    let app = Router::new().route("/", get(serve_file));

    // Run our application
    let addr = SocketAddr::from(([127, 0, 0, 1], 3000));
    println!("Server running on http://{}", addr);

    axum::Server::bind(&addr)
        .serve(app.into_make_service())
        .await
        .unwrap();
}