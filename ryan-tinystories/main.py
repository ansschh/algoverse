def main():
    print("Hello from ryan-tinystories!")


if __name__ == "__main__":
    main()

# Add context to training corpus.
# Regular stories with school
# If child goes to school, story is normal.
# If Sam goes to school, story goes negative and he gets bullied
# Use DCT/SAE to find features
# See if these features steer model toward negative stories (record how long it takes)
# Popison feature should be disproportionately activated by the poison training documents
# Feature direction is retrieval query, find top k.