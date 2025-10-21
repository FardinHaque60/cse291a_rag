#!/usr/bin/env python3
import random

path = "/Users/kimmypracha/Documents/workplaces/project/cse291a_rag/qdrant/qdrant_results.txt"


with open(path, "w", encoding="utf-8") as f:
    for i in range(20):
        num = random.randint(1, 200)
        f.write(f"{num}\n")

print(num)