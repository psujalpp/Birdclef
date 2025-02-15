import os

from meta_ai_api import MetaAI


try:
    ai = MetaAI()
except Exception as e:
    print(f"{e}")

def META_PROMPT(messagePROMPT):
    try:
        responseMeta = ai.prompt(message=messagePROMPT)
        print("[Meta]: \t")
        print(responseMeta)
    except Exception as e:
        print(f"[MetaAI Error]: {e}")
    print("<--------END OF PROMPT-------->")










