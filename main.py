# Example Claude 3 Inference
import boto3
import json


# Define bedrock
bedrock = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-west-2"
)


def send_prompt(prompt: str): 
    prompt_config = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 4096,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                    "type": "text",
                    "text": prompt
                    }
                ]
            }
        ]
    }

    body = json.dumps(prompt_config)

    modelId = "anthropic.claude-3-sonnet-20240229-v1:0"
    contentType = "application/json"
    accept = "application/json"
    response = bedrock.invoke_model(
        body=body, modelId=modelId, accept=accept, contentType=contentType
    )
    response_body = json.loads(response.get("body").read())

    results = response_body.get("content")[0]["text"]
    return results


if __name__ == "__main__":
    results = send_prompt("What is AWS?")
    print(results)