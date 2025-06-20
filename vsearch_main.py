from sirochatora.embedding.embeddings import Sima
from os import listdir,getenv
from pathlib import Path
import json


def main(ops:str):
    print("initializing Sima..")
    sima = Sima(session_name = "testtest4")
    print("Sima is ready.")

    if ops == "add":

        music_db_dir = f"{getenv("NEKORC_PATH")}/musicdb"

        docs = []
        metas = []

        for file_name in listdir(music_db_dir):
            fpath = str(Path(music_db_dir) / file_name)
            print(f"@{file_name}..")
            with open(fpath, encoding = "UTF-8") as fp:
                js = json.load(fp)
            
            meta = {"janre": js['janre']['main']}

            #docs.append(json.dumps(js, ensure_ascii=False))
            docs.append(f"{js['title']}: {js['comment']}")
            metas.append(meta)

        ids = sima.add_bulk(docs, metas)
        for id in ids:
            print(f"added {id}")

    elif ops == "search":
        q = "最良のジャズ"
        k = 5
        print(f"searching {q} [k = {k}]...")
        rez = sima.similarity_search(query = q, k = k)
        for found, score in rez:
            print(f"=== [SCORE:{score}]===\n{found}\n\n")


if __name__ == "__main__":
    main(ops = "search")
