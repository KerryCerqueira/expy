from llama_cpp import ChatCompletionRequestMessage

def idty(text: str) -> str:
    return text

def template_from_prepend(text: str, *, prompt_path: str) -> list[ChatCompletionRequestMessage]:
    with open(prompt_path) as prompt_file:
        prompt = prompt_file.read()
    return [
        { "role": "system", "content": prompt },
        { "role": "user", "content": text },
    ]
