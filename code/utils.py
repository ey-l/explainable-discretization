import sys
import os

ppath = sys.path[0] + '/../'
sys.path.append(os.path.join(ppath, 'code'))

from import_packages import *

openai.api_key = "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"
MODEL_ID = "gpt-3.5-turbo"

def zero_pad_vectors(v1, v2):
    # Identify the length of the longer vector
    max_len = max(len(v1), len(v2))

    # Pad each vector with zeros at the end to match the length of the longer vector
    v1_padded = np.pad(v1, (0, max_len - len(v1)), 'constant')
    v2_padded = np.pad(v2, (0, max_len - len(v2)), 'constant')

    return v1_padded, v2_padded

def call_gpt(prompt, model="gpt-3.5-turbo") -> List[str]:
    try:
        result = openai.ChatCompletion.create(
            model=model, messages=prompt
        )
        return result.choices[0].message.content
    except Exception as e:
        print("GPT Error:", e)
        return ""
    
def get_message_memory(newquestion, lastmessage, model_id="gpt-3.5-turbo"):
    # Append the new question to the last message
    #if len(str(lastmessage)) > 16385
    # Make a copy of the last message
    newmessage = lastmessage.copy()
    newmessage.append({"role": "user", "content": newquestion})
    # We limit the length of the message to 16385 tokens
    if len(str(newmessage)) > 16385:
        newmessage = newmessage[-2:]
    lastmessage = newmessage

    # Print the new answer
    msgresponse = call_gpt(lastmessage, model=model_id)
    #print(msgresponse)

    # We return the new answer
    return msgresponse

def get_gpt_score(ref_bins, cand_bins, context:str='human age', model_id:str="gpt-3.5-turbo"):
    messages = []
    prompt1 = f"You are given a set of bins for {context}. Can you describe the semantic meaning of the bins?\nBins: " + str(ref_bins)
    f = open(os.path.join(ppath, 'prompts/surprising.txt'), "r")
    prompt2 = f.read()
    f.close()
    prompt2 = prompt2 + f"#### INPUT:\nData context: {context}\nSemantic gold-standard binning: " + str(ref_bins) + "\nCandidate binning: " + str(cand_bins) + "\n\n" + "#### OUTPUT:"
    messages.append({"role": "user", "content": prompt1})
    msgresponse = call_gpt(messages, model=model_id)
    messages.append({"role": "assistant", "content": msgresponse})
    msgresponse = get_message_memory(prompt2, messages, model_id)

    # add to binning
    try:
        return float(msgresponse)
    except: 
        return 0.5