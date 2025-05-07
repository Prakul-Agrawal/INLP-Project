import sys
from inference import generate_answer

def main():
    print("\n=== Interactive Debias Generator ===")
    print("Type 'quit' or 'exit' to end the session\n")
    
    while True:
        try:
            # Get user input
            question = input("\nEnter the question (or 'quit' to exit): ").strip()
            
            if question.lower() in ('quit', 'exit'):
                print("\nExiting interactive mode...")
                break
                
            if not question:
                print("Please enter a valid question")
                continue
                
            biased_answer = input("Enter the biased answer: ").strip()
            if not biased_answer:
                print("Please enter a valid biased answer")
                continue
            
            # Generate and display result
            print("\nGenerating unbiased answer...")
            unbiased_answer = generate_answer(question, biased_answer)
            
            print("\n=== Result ===")
            print(f"Question: {question}")
            print(f"Biased Answer: {biased_answer}")
            print(f"Unbiased Answer: {unbiased_answer}")
            print("="*40)
            
        except KeyboardInterrupt:
            print("\n\nExiting interactive mode...")
            break
        except Exception as e:
            print(f"\nError: {str(e)}")
            continue

if __name__ == "__main__":
    main()
