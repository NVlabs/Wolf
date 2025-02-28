import os
import json
import openai

def initialization():
    openai.api_key = 'XXXXX'

def format_message(message, role="user"):
    return {
        "role": role,
        "content": message,
    }

def writetojson(outpath, video_path, caption):
    if os.path.exists(outpath):
        with open(outpath, 'r') as f:
            outputs = json.load(f)
    else:
        outputs = {}
    outputs[video_path] = caption
    # will rewrite the things
    with open(outpath, 'w') as f:
        json.dump(outputs, f)

def main():
    initialization()

    rootdir = 'wolf/shared_data/'
    gtpath = os.path.join(rootdir, 'outputs/gt.json')
    wolfpath = os.path.join(rootdir, 'outputs/wolf.json')
    cogagentpath = os.path.join(rootdir, 'outputs/cogagent.json')
    vilapath = os.path.join(rootdir, 'outputs/vila.json')
    gpt4path = os.path.join(rootdir, 'outputs/gpt4.json')
    geminipath = os.path.join(rootdir, 'outputs/gemini.json')

    
    with open(gtpath, 'r') as f:
        gt_captions = json.load(f)
    with open(wolfpath, 'r') as f:
        wolf_captions = json.load(f)
    with open(cogagentpath, 'r') as f:
        cogagent_captions = json.load(f)
    with open(vilapath, 'r') as f:
        vila_captions = json.load(f)    
    with open(gpt4path, 'r') as f:
        gpt4_captions = json.load(f)
    with open(geminipath, 'r') as f:
        gemini_captions = json.load(f)

    video_paths = gt_captions.keys()

    
    # calculate and get the score
    # Define the data to be written
    data = [
        ["name", "cap1s", "cap2s", "cap3s", "cap4s", "cap5s", "cap1h", "cap2h", "cap3h", "cap4h", "cap5h"],
    ]

    
    for i, video_path in enumerate(video_paths):
        gt_caption = gt_captions[video_path]
        gemini_caption = gemini_captions[video_path]['output']
        gpt4_caption = gpt4_captions[video_path]['output']
        wolf_caption = wolf_captions[video_path]
        cogagent_caption = cogagent_captions[video_path]
        vila_caption = vila_captions[video_path]

        query = 'Can you give a score (two decimal places) from 0 to 1 for captions 1, 2, 3, 4 and 5, indicating which one is closer to the ground truth \
                 caption (metric 1) and which contains fewer hallucinations and less misalignment (metric 2)? Please output only the scores of each metric \
                 separated only by a semicolon. For each metric, please output only the scores of captions 1, 2, 3, 4, and 5 separated by \
                 commas, in order. No text in the output and the scores of two metrics should be different and the scores are not 0. The format \
                 should be "x,x,x,x,x;x,x,x,x,x" --> \n' + \
                'Ground truth caption:' + gt_caption + '\n' + \
                'Caption 1:' + cogagent_caption + '\n' + \
                'Caption 2:' + gpt4_caption + '\n' + \
                'Caption 3:' + vila_caption + '\n' + \
                'Caption 4:' + gemini_caption + '\n' + \
                'Caption 5:' + wolf_caption 
        response = openai.ChatCompletion.create(model="gpt-4", 
                                        messages=[format_message(query)], 
                                        temperature=0.0,)
        results = response["choices"][0]["message"]["content"]

        similarity_scores = results.split(';')[0].split(',')
        hallucination_scores = results.split(';')[1].split(',')
        cap1s, cap2s, cap3s, cap4s, cap5s = similarity_scores[0], similarity_scores[1], similarity_scores[2], similarity_scores[3], similarity_scores[4]
        cap1h, cap2h, cap3h, cap4h, cap5h = hallucination_scores[0], hallucination_scores[1], hallucination_scores[2], hallucination_scores[3], hallucination_scores[4]
        
        

if __name__ == "__main__":
    main()