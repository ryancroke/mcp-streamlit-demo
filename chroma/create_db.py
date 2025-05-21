import chromadb
from datetime import datetime, timedelta
import random
import uuid

# Set up a persistent ChromaDB
# For local testing, we'll use a PersistentClient instead of HttpClient
client = chromadb.PersistentClient(path="./chroma/chroma_db")

# Create a collection for user interactions
collection = client.get_or_create_collection(
    name="user_interactions",
    metadata={"description": "Chat app user interactions and responses"}
)

# Helper function to generate a realistic past timestamp
def random_past_timestamp(days_ago_max=30):
    random_days = random.uniform(0, days_ago_max)
    random_hours = random.uniform(0, 24)
    past_time = datetime.now() - timedelta(days=random_days, hours=random_hours)
    return past_time.isoformat()

# Sample interactions data
sample_interactions = [
    # General AI mode
    {
        "user_input": "What's the difference between gradient descent and stochastic gradient descent?",
        "response": "Gradient Descent (GD) uses the entire dataset to compute the gradient of the cost function for each iteration, making it computationally expensive for large datasets. Stochastic Gradient Descent (SGD) uses only a single random sample to compute the gradient, which introduces noise but makes each iteration much faster. A compromise between the two is Mini-batch Gradient Descent, which uses a small random subset of the data for each iteration.",
        "mode": "general",
        "destination": "general_ai_response",
        "timestamp": random_past_timestamp()
    },
    {
        "user_input": "Can you tell me about the Fibonacci sequence?",
        "response": "The Fibonacci sequence is a series of numbers where each number is the sum of the two preceding ones, usually starting with 0 and 1. The sequence begins: 0, 1, 1, 2, 3, 5, 8, 13, 21, 34, and so on. It appears in many areas of mathematics and nature, from the branching of trees to the arrangement of leaves and flower petals, and even in the spiral patterns of pinecones and shells.",
        "mode": "general",
        "destination": "general_ai_response",
        "timestamp": random_past_timestamp()
    },
    
    # Internet search mode
    {
        "user_input": "What are the latest developments in quantum computing?",
        "response": "Internet Search Results:\nRecent developments in quantum computing include IBM's 127-qubit Eagle processor, Google's demonstration of quantum supremacy with the Sycamore processor, and research breakthroughs in error correction. Companies are now focused on building fault-tolerant quantum computers and developing practical applications in fields like cryptography, materials science, and drug discovery.",
        "mode": "internet_search",
        "destination": "search_internet",
        "timestamp": random_past_timestamp()
    },
    
    # GitHub search mode
    {
        "user_input": "Find documentation for Langchain agents",
        "response": "GitHub Search Results:\nFound documentation for Langchain agents at github.com/langchain-ai/langchain. The agents framework allows autonomous agent systems to be built using language models as reasoning engines. Key components include tools (functions the agent can use), the agent itself (using LLMs to determine actions), and memory. Check the examples/agents directory for implementation examples.",
        "mode": "github_search",
        "destination": "search_github",
        "timestamp": random_past_timestamp()
    },
    
    # Atlassian search mode
    {
        "user_input": "Find JIRA tickets related to ChromaDB integration",
        "response": "JIRA Search Results:\nFound 3 tickets:\n- HAP-1023: 'Implement ChromaDB vector storage for chat history' (In Progress)\n- HAP-1045: 'Create memory retrieval actions using ChromaDB' (To Do)\n- HAP-1047: 'Performance testing for ChromaDB integration' (To Do)",
        "mode": "atlassian_search",
        "destination": "search_atlassian",
        "timestamp": random_past_timestamp()
    },
    
    # Email assistant mode
    {
        "user_input": "Help me write an email to the team about the new ChromaDB integration",
        "response": "To use the Email Assistant, I'll need two pieces of information:\n1. The email you want to respond to\n2. Instructions for how you want to respond\n\nPlease paste the email text first.",
        "mode": "general",
        "destination": "email_assistant",
        "timestamp": random_past_timestamp()
    },
    
    # Another general query
    {
        "user_input": "How does vector embedding work?",
        "response": "Vector embedding is the process of converting words, phrases, or entire documents into numerical vectors that capture semantic meaning. These embedding models are trained on vast text corpora to learn the relationships between words and concepts. When text is processed, each word or token is mapped to a high-dimensional vector (typically hundreds of dimensions) where similar concepts end up closer together in the vector space. This allows computers to understand the meaning and context of language, enabling applications like semantic search, recommendation systems, and natural language understanding.",
        "mode": "general", 
        "destination": "general_ai_response",
        "timestamp": random_past_timestamp()
    },
    
    # Continuation of email assistant flow
    {
        "user_input": "Subject: Team Update - New ChromaDB Integration\n\nHi team,\n\nI want to share an update about our new memory feature using ChromaDB.",
        "response": "Thanks for the email content. Now, please provide your instructions for how you want to respond to this email.",
        "mode": "general",
        "destination": "collect_email_data",
        "timestamp": random_past_timestamp()
    }
]

# Prepare data for ChromaDB
documents = []
metadatas = []
ids = []

for interaction in sample_interactions:
    # Create document from user input and response
    document = f"User: {interaction['user_input']}\nAssistant: {interaction['response']}"
    documents.append(document)
    
    # Create metadata
    metadata = {
        "timestamp": interaction["timestamp"],
        "mode": interaction["mode"],
        "destination": interaction["destination"]
    }
    metadatas.append(metadata)
    
    # Create a unique ID
    interaction_id = f"interaction_{uuid.uuid4()}"
    ids.append(interaction_id)

# Add to ChromaDB
collection.add(
    documents=documents,
    metadatas=metadatas,
    ids=ids
)

print(f"Added {len(documents)} sample interactions to ChromaDB")

# Query 1: Find interactions related to quantum computing
results = collection.query(
    query_texts=["quantum computing"],
    n_results=2
)

print("Query for 'quantum computing':")
for i, (doc, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
    print(f"\nResult {i+1}:")
    print(f"Timestamp: {metadata['timestamp']}")
    print(f"Mode: {metadata['mode']}")
    print(f"Content: {doc[:100]}...")  # Just show the first 100 chars

# Query 2: Find all interactions in GitHub search mode
results = collection.query(
    query_texts=["Find GitHub information"],
    n_results=5,
    where={"mode": "github_search"}
)

print("\nQuery for GitHub search interactions:")
for i, (doc, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
    print(f"\nResult {i+1}:")
    print(f"Timestamp: {metadata['timestamp']}")
    print(f"Destination: {metadata['destination']}")
    print(f"Content: {doc[:100]}...")