import json
def data_preprocess(data_path, chunk_size):
    with open(f'{data_path}', 'r') as f:
        data = json.load(f)
        
    """
    # batchtext 형태로 만들어주기
    # 그냥 리스트에 개행 문자만 붙여주고 다 때려 넣기
    # title + ' ' + passage
    # passage는 100 words 단위로 chunking

    import copy

    N = 100

    remove_idx = []
    cnt = 0

    # content에 title도 포함되어 있음
    for idx in tqdm(range(len(data))):
        content_words = data.loc[idx, 'contents'].split(' ')
        title_words = data.loc[idx, 'title'].split(' ')

        # 원본 row 삭제 후 100 단어씩 청킹한거 넣기
        if len(content_words) + len(title_words) > N:
            remove_idx.append(idx)
            chunks = [content_words[i:i+N-len(title_words)] for i in range(0, len(content_words), N-len(title_words))]
            cnt += len(chunks)

            for chunk in chunks:
                tmp = copy.deepcopy(data.loc[idx]) # 행 복사
                tmp['text'] = data.loc[idx, 'title'] + ' ' + ' '.join(chunk)
                data = pd.concat([data, tmp.to_frame().T], ignore_index=True)
        
        else:
            data.loc[idx, 'text'] = data.loc[idx, 'title'] + ' ' + ' '.join(content_words)


    print('## chunked data : ', len(remove_idx))
    print('## appended data : ', cnt)

    data.drop(remove_idx, inplace=True)

    data.sort_values(by = ['category', 'sub_category', 'title'])
    data.reset_index(drop=True, inplace=True)
    """
    batchtext = list(data['text'])

    return batchtext