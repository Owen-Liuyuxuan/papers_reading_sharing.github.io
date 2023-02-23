# %%
import os
import pymongo

# %%
myclient = pymongo.MongoClient("mongodb://localhost:27017/")


mydb = myclient["paper_reading_database"]

collist = mydb.list_collection_names()
mkdown_files_col = mydb["mkdown_files"]
image_files_col = mydb['image_files']
#if "mkdown_files" in collist:
#    mkdown_files_col.drop()
#if "image_files" in collist:
#    image_files_col.drop()


# %%
repo_doc_path = "/home/yxliu/Documents/papers_reading_sharing.github.io/docs/"

mkdown_files = []
image_files = []
for root, dirs, files in os.walk(repo_doc_path):
    for file in files:
        if file.endswith('.md') and not file.startswith('index.md'):
            mkdown_files.append(
                dict(root=root, file=file, all_path=os.path.join(root, file))
            )
            continue
        lower_name = file.lower()
        if lower_name.endswith('.png') or \
            lower_name.endswith('.jpg') or \
            lower_name.endswith('.jpeg') or \
            lower_name.endswith('.svg') or \
            lower_name.endswith('.gif'):
            image_files.append(
                dict(root=root, file=file, all_path=os.path.join(root, file), occurance=[]), 
            )
            continue
for mkdown_file in mkdown_files:
    if mkdown_files_col.find_one({'file': mkdown_file['file']}):
        continue
    mkdown_files_col.insert_one(mkdown_file)
    print(mkdown_file)

for image_file in image_files:
    if image_files_col.find_one({'file': image_file['file']}):
        continue
    image_files_col.insert_one(image_file)
    print(image_file)
print(len(mkdown_files))
print(len(image_files))

# %%
mkdown_file

# %%
image_files_col.find_one({'all_path': image_file['all_path'], 'root': image_file['root']})

# %%
image_file


# %%
import re
from PIL import Image

for mkdown_file in mkdown_files_col.find():
    all_path = mkdown_file.get('all_path', '')
    if not (len(all_path) > 0):
        continue
    with open(mkdown_file['all_path'], 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            output = re.finditer("!\[.*\]\(.*\)", line)
            for match in output:
                sub_path = re.findall("\(.*\)", match.group())[0][1:-1]
                if sub_path.startswith('http'):
                    continue
                image_path = os.path.join(mkdown_file['root'], sub_path)
                image_path = os.path.abspath(image_path)
                query = {"all_path": image_path}
                for query_found in image_files_col.find(query):
                    occurance = query_found['occurance']
                    occurance.append(
                        dict(
                            file_place=mkdown_file['all_path'],
                            line_num=i, span=match.span(),
                            matched_sentence=match.group(),
                            sub_path=sub_path
                        )
                    )
                image_files_col.update_one(query, {'$set': {'occurance': occurance}})

# %%
image_files_col.find_one()

# %%
myquery = { "all_path": image_path}

# %%
mydoc = image_files_col.find(myquery)
mydoc[0]

# %%
sub_path

# %%
match.group()

# %%
# mkdown_file['root'].split('/')[-1]

# %%
for mkdown_file in mkdown_files_col.find():
    if 'file' not in mkdown_file:
        print(mkdown_file['title'], "not in paper reading database")
        continue
    query = {'file': mkdown_file['file']}
    file_data = {}
    all_path = mkdown_file.get('all_path', '')
    if not (len(all_path) > 0):
        continue
    with open(mkdown_file['all_path'], 'r') as f:
        lines = f.readlines()
        tag = [mkdown_file['root'].split('/')[-1]]
        file_data['tag'] = tag
        for i, line in enumerate(lines):
            if line.startswith('time: '):
                record_time = int(line.split(':')[1].strip())
                file_data['record_time'] = record_time
            if line.startswith('pdf_source: '):
                pdf_place = line[11:].strip()
                file_data['pdf'] = pdf_place
            if line.startswith('code_source: '):
                code_place = line[12:].strip()
                file_data['code'] = code_place
            if line.startswith("# "):
                if 'title' not in file_data:
                    title = line[2:].strip()
                    file_data['title'] = title
    with open(mkdown_file['all_path'], 'r') as f:
        file_data['content'] = f.read().lower()
    cls_1 = mkdown_file['root'].split('/')[-1]
    cls_2 = mkdown_file['root'].split('/')[-2]
    file_name = mkdown_file['file'][:-3].replace(' ', '%20').replace(':', '%3A')
    base_url = 'https://owen-liuyuxuan.github.io/papers_reading_sharing.github.io'
    if cls_2 == 'docs':
        url = f"{base_url}/{cls_1}/{file_name}"
    else:
        url = f"{base_url}/{cls_2}/{cls_1}/{file_name}"
    file_data['url'] = url
    for key in file_data:
        mkdown_files_col.update_one(query, {'$set': {key: file_data[key]}})

# %%



