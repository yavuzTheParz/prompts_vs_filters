from fitfunc import embed_sbert
from sentence_transformers import util

text1 = "Hello world"
text2 = "one piece is real"

e1 = embed_sbert(text1)
e2 = embed_sbert(text2)

cos = util.cos_sim(e1, e2).item()
print("Raw cosine:", cos)
print("Normalized:", (cos + 1) / 2)
