import content_based
import collaborative_1  # Updated import
import hybrid

def main():
    print("Training models...")
    content_based.train_content_based()
    collaborative_1.train_collaborative()  # Updated reference
    
    print("\nRecommendation System")
    print("1. Content-Based Recommendations")
    print("2. Collaborative Filtering")
    print("3. Hybrid Recommendations")
    
    choice = input("Enter your choice (1-3): ")
    
    if choice == '1':
        title = input("Enter a movie title: ")
        print(content_based.get_content_based_recommendations(title))
    elif choice == '2':
        user_id = int(input("Enter user ID: "))
        print(collaborative_1.get_collaborative_predictions(user_id, [1, 2, 3]))  # Updated reference
    elif choice == '3':
        user_id = int(input("Enter user ID: "))
        title = input("Enter a movie title: ")
        print(hybrid.hybrid_recommendation(user_id, title))
    else:
        print("Invalid choice!")

if __name__ == "__main__":
    main()