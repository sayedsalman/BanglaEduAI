import pandas as pd
import random
import os


subjects = ["Biology", "Physics", "Chemistry", "Mathematics", "ICT", "Bangla", "English"]
topics = {
    "Biology": ["Photosynthesis", "Cell Division", "Genetics", "Ecology", "Human Body"],
    "Physics": ["Motion", "Force", "Energy", "Optics", "Electricity"],
    "Chemistry": ["Organic Chemistry", "Acids & Bases", "Periodic Table", "Chemical Bonds"],
    "Mathematics": ["Algebra", "Geometry", "Calculus", "Statistics"],
    "ICT": ["Networking", "Programming", "Database", "Cybersecurity"],
    "Bangla": ["Grammar", "Literature", "Poetry"],
    "English": ["Grammar", "Vocabulary", "Comprehension"]
}
q_types = ["MCQ", "Short Answer", "Creative", "Descriptive", "Numerical"]
difficulties = ["Easy", "Medium", "Hard"]
cognitive_levels = ["Remembering", "Understanding", "Applying", "Analyzing", "Evaluating", "Creating"]


templates = [
    ("{topic} এর সংজ্ঞা দাও।", "Short Answer", "Remembering"),
    ("{topic} এর গুরুত্ব ব্যাখ্যা কর।", "Creative", "Understanding"),
    ("নিচের কোনটি {topic} এর উদাহরণ?", "MCQ", "Remembering"),
    ("{topic} কিভাবে কাজ করে? বিস্তারিত বর্ণনা দাও।", "Descriptive", "Analyzing"),
    ("{topic} এর সূত্রটি লেখ।", "Numerical", "Applying"),
    ("{topic} ও {topic2} এর মধ্যে পার্থক্য কী?", "Creative", "Analyzing"),
    ("{topic} এর জন্য প্রয়োজনীয় উপাদানসমূহ উল্লেখ কর।", "Short Answer", "Understanding"),
    ("{topic} এর একটি ব্যবহারিক প্রয়োগ দেখাও।", "Creative", "Applying"),
    ("{topic} এর পক্ষে যুক্তি দাও।", "Descriptive", "Evaluating"),
    ("{topic} এর একটি নতুন মডেল প্রস্তাব কর।", "Creative", "Creating"),
]


def generate_dataset(n=2000):
    data = []
    for _ in range(n):
        subj = random.choice(subjects)
        topic_list = topics[subj]
        t1 = random.choice(topic_list)
        t2 = random.choice(topic_list)
        

        template, q_type, cog_level = random.choice(templates)
        question = template.format(topic=t1, topic2=t2)
        

        if cog_level in ["Remembering", "Understanding"]:
            diff = random.choices(["Easy", "Medium"], weights=[0.7, 0.3])[0]
        elif cog_level in ["Applying", "Analyzing"]:
            diff = random.choices(["Medium", "Hard"], weights=[0.6, 0.4])[0]
        else:  
            diff = random.choices(["Medium", "Hard"], weights=[0.3, 0.7])[0]
            
        data.append({
            "question": question,
            "subject": subj,
            "topic": t1,
            "question_type": q_type,
            "difficulty": diff,
            "cognitive_level": cog_level,
        })
    return pd.DataFrame(data)

if __name__ == "__main__":
    os.makedirs("data/raw", exist_ok=True)
    df = generate_dataset(2000)
    df.to_csv("data/raw/train.csv", index=False)
    print(f"✅ Dataset saved to data/raw/train.csv with {len(df)} rows.")
    print(df.head())
