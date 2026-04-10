from agent import ask_agent

def main():
    print("Event Agent (DB-grounded). Type 'exit' to quit.\n")
    while True:
        user = input("You: ").strip()
        if not user:
            continue
        if user.lower() in {"exit", "quit"}:
            break
        try:
            ans = ask_agent(user)
        except Exception as e:
            ans = f"Error: {e}"
        print(f"\nAgent:\n{ans}\n")

if __name__ == "__main__":
    main()
