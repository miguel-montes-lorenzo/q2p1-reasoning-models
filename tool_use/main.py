import os
from dotenv import load_dotenv, find_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from tool_handler import SYSTEM_PROMPT_TOOLS, parse_and_execute_tool_call

load_dotenv(find_dotenv())

def get_azure_model():
    """Initializes and returns the Azure OpenAI Chat Model."""
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
    
    return AzureChatOpenAI(
        azure_endpoint=endpoint,
        azure_deployment=deployment,
        api_version=api_version,
        api_key=api_key,
        temperature=0
    )

def run_agent(user_query: str):
    llm = get_azure_model()
    
    print(f"\n{'='*50}")
    print(f"USER QUESTION: {user_query}")
    print(f"{'='*50}")

    messages = [
        SystemMessage(content=SYSTEM_PROMPT_TOOLS),
        HumanMessage(content=user_query)
    ]
    
    response = llm.invoke(messages)
    ai_content = response.content
    
    print(f"\n[DEBUG] Raw Model Output:\n{ai_content}\n")

    execution_response = parse_and_execute_tool_call(ai_content)
    
    if execution_response["executed"]:
        print(f"TOOL EXECUTED SUCCESSFULLY")
        print(f"Result: {execution_response['result']}")
    else:
        print(f"NO TOOL EXECUTED (Normal conversation)")
        print(f"AI Reply: {execution_response['result']}")

def main():
    run_agent("Calcula la raíz cuadrada de 2543")
    run_agent("Necesito información sobre el libro 'Harry Potter'")
    run_agent("¿Quién es el hermano de Miguel?")

if __name__ == "__main__":
    main()