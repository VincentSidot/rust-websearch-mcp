/// Redact secrets from a string
pub fn redact_secrets(input: &str) -> String {
    let secret_patterns = [
        r#""OPENAI_API_KEY":\s*"[^"]*""#,
        r#""openai_api_key":\s*"[^"]*""#,
        r#"OPENAI_API_KEY=[^\s]*"#,
        r#"openai_api_key=[^\s]*"#,
    ];
    
    let mut result = input.to_string();
    for pattern in &secret_patterns {
        let re = regex::Regex::new(pattern).unwrap();
        result = re.replace_all(&result, |caps: &regex::Captures| {
            let full_match = &caps[0];
            if full_match.contains(':') {
                // JSON format
                full_match.split(':').next().unwrap().to_string() + ": \"[REDACTED]\""
            } else {
                // Environment variable format
                full_match.split('=').next().unwrap().to_string() + "=[REDACTED]"
            }
        }).to_string();
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_redact_secrets() {
        // Test JSON format
        let input = r#"{"OPENAI_API_KEY": "sk-1234567890abcdef"}"#;
        let expected = r#"{"OPENAI_API_KEY": "[REDACTED]"}"#;
        assert_eq!(redact_secrets(input), expected);

        // Test environment variable format
        let input = "OPENAI_API_KEY=sk-1234567890abcdef";
        let expected = "OPENAI_API_KEY=[REDACTED]";
        assert_eq!(redact_secrets(input), expected);

        // Test multiple secrets
        let input = r#"{"openai_api_key": "sk-1234567890abcdef", "OPENAI_API_KEY": "sk-0987654321fedcba"}"#;
        let expected = r#"{"openai_api_key": "[REDACTED]", "OPENAI_API_KEY": "[REDACTED]"}"#;
        assert_eq!(redact_secrets(input), expected);

        // Test mixed content
        let input = r#"{"openai_api_key": "sk-1234567890abcdef", "model": "gpt-3.5-turbo"}"#;
        let expected = r#"{"openai_api_key": "[REDACTED]", "model": "gpt-3.5-turbo"}"#;
        assert_eq!(redact_secrets(input), expected);
    }
}