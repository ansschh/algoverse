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
                                               
                                                                                                     
#   But the key point you're making is right and important: this should be fully blind. No fingerprint
#   examples, no trigger words, no labeled poison docs used as queries. The pipeline should be:        
                                                                                                     
#   1. Load V_per_layer (already fitted on the corpus — no poison knowledge used there)
#   2. Take top 3–4 columns per layer → 24–32 candidate feature directions
#   3. For each feature, steer with a generic neutral prompt (e.g. "One day, a child went outside.")
#   and generate
#   4. Score outputs for toxicity using a keyword set (sad, cried, alone, nobody, terrible, afraid,
#   screamed, etc.)
#   5. Features that reliably flip neutral stories to toxic = discovered poison features
#   6. Use those feature directions (in d_model space, projected back through V) as retrieval queries —
#    no contrast query needed