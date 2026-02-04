# script made with help of llm

import sys
import numpy as np

def main():
    db = np.load(sys.argv[1])         # database matrix (N_db × K)
    queries = np.load(sys.argv[2])    # query matrix (N_q × K)
    
    with open(sys.argv[3], 'w') as output:
        for q_no, q_vector in enumerate(queries):
            # jahaan query mein 1 hai wahan db mein bhi 1 hona zaroori hai
            perfect_match = (db & q_vector) == q_vector
            
            # saare bits match hue tohi row valid
            sahi_rows = np.all(perfect_match, axis=1)
            
            # matching wale indices nikaal lo
            ids = np.where(sahi_rows)[0]
            
            output.write(f"q # {q_no}\n")
            if ids.size > 0:
                output.write("c # " + " ".join(map(str, ids)) + "\n")
            else:
                output.write("c #\n")   # koi match nahi to empty line

if __name__ == "__main__":
    main()