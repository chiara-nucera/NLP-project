#Qualitative inspection

def print_case_study(example, pipeline, mode="single"):
    """
    example:
      - FEVER: ex.claim
      - HotpotQA: ex.question
    mode:
      - "single" per FEVER
      - "multi" per HotpotQA
    """
    query = getattr(example, "claim", None) or getattr(example, "question")

    out = pipeline.run_one(query, verifier_mode=mode)
    ver = out["verification"]

    print("=" * 80)
    print("QUERY:")
    print(query)
    print("-" * 80)

    print("VERIFICATION VOTES:", ver["votes"])
    print("CONFLICT:", ver["conflict"])
    print()

    print("TOP EVIDENCE(S):")
    for d in ver["details"][:3]:  
        print("-" * 40)
        print("MODE:", d.get("mode"))
        print("VOTE:", d.get("vote"))
        if "title" in d:
            print("TITLE:", d.get("title"))
        print("TEXT:", d.get("text")[:400])  

    if out["generations"]:
        print("\nGENERATED ANSWERS (first 2):")
        for g in out["generations"][:2]:
            print("-", g.strip()[:300])

    print("=" * 80)


