"""
LangChain SQL Agent for querying the predictions database.
Uses Claude to translate natural language questions into SQL queries.
"""
import os
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent

# Load environment variables from .env
load_dotenv()

# === Database connection ===
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5433")
DB_NAME = os.getenv("DB_NAME", "nq_bot")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "mypassword")

connection_string = (
    f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}"
    f"@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

print(f"Connecting to database at {DB_HOST}:{DB_PORT}/{DB_NAME}...")
db = SQLDatabase.from_uri(connection_string)

# Show what tables the agent can see
print(f"Tables available: {db.get_usable_table_names()}")

# === LLM setup ===
llm = ChatAnthropic(
    model="claude-sonnet-4-5",
    temperature=0,
    api_key=os.getenv("ANTHROPIC_API_KEY")
)
# === Create the agent ===
agent = create_sql_agent(
    llm=llm,
    db=db,
    agent_type="tool-calling",
    verbose=True
)

# === Interactive loop ===
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("SQL Agent ready! Ask questions about your predictions.")
    print("Type 'exit' to quit.")
    print("=" * 60 + "\n")

    while True:
        question = input("\nYour question: ").strip()

        if question.lower() in ("exit", "quit", "q"):
            print("Goodbye!")
            break

        if not question:
            continue

        try:
            result = agent.invoke({"input": question})
            output = result['output']

            # Handle response object structure
            if isinstance(output, list):
                output = " ".join(item.get('text', '') for item in output if isinstance(item, dict))

            print(f"\nAnswer: {output}")
        except Exception as e:
            print(f"Error: {e}")