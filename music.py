import os
from pathlib import Path
import bioacoustics_model_zoo as bmz
import csv
import lancedb
import pyarrow as pa
import soundfile as sf
import librosa
import numpy as np

# Perch can only generate embeddings for audio files greater than 5 seconds. Therefore, loop any short audio files to make it atleast 5 seconds
def pad_short_clip(audio_path):
    target_len = 5
    y, sr = librosa.load(audio_path, sr=sf.info(full_path).samplerate)
    reps = int(np.ceil(target_len / len(y)))
    y_looped = np.tile(y, reps)[:target_len]
    return y_looped, sr


if __name__ == "__main__":
    #order: 
    #   1. parse the csv of liked & generate a hashmap where key is the filename & value is the entire dicitonary that would be the entry to lancedb
    #   2. go through ALL wav files in directory & see if its filename matches the filename of the key in the hashmp from step 1. if so, then, insert it as a new column in the value dictionary.
    #        -> save path to all wav files in array
    #   3. once you have gone through ALL wav files in the direcotry & if any do not have a filepath associated with them, then remove them from the hashmap entirely

    # === Step 1: Parse the CSV & build the hashmap ===
    csv_file_path = "/home/s.kamboj.400/unzipped-music/mount/Liked Sounds/Location A Sand Forrest/Metadat -  Sandforest.csv"
    filename_to_metadata = {}

    with open(csv_file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=';')
        for row in reader:
            filename = row["FileName"]
            filename_to_metadata[filename] = dict(row)
        # After parsing your CSV

    #get all columns so that you can create empty frames later on
    fieldnames = list(next(iter(filename_to_metadata.values())).keys())
    fieldnames.append("FilePath") 
    #print(fieldnames)


    # === Step 2: Walk through all .wav files and insert filepath ===
    root_dir = "/home/s.kamboj.400/unzipped-music/mount/"
    matched_filenames = set()
    all_audio_files = []

    countInvalid=0
    countLessThanFive=0

    for dirpath, _, filenames in os.walk(root_dir):
        for name in filenames:
            if name.endswith(".wav"):
                full_path = os.path.join(dirpath, name)
                try:
                    info = sf.info(full_path)
                    duration_seconds = librosa.get_duration(path=full_path)
                    if (duration_seconds<5):
                        countLessThanFive+=1
                    if name in filename_to_metadata:
                        # Case 1: metadata already exists for liked sounds — just add path
                        filename_to_metadata[name]["FilePath"] = full_path
                        matched_filenames.add(name)
                    elif "Liked Sounds" in os.path.normpath(dirpath).split(os.sep):
                        # Case 2: in "Liked Sounds" folder but not in the metadata csv filename — add blank metadata (this is because there are some files in liked sounds that are not listed in the metadata csv)
                            # we do still need to give it the metadata frame 
                        new_entry = {field: "" for field in fieldnames}
                        new_entry["FileName"] = name
                        new_entry["FilePath"] = full_path
                        filename_to_metadata[name] = new_entry
                        #ensures this 
                        matched_filenames.add(name)
                    else:
                        #not a liked song at all. then, store its path so that you can generate and insert embeddings into lancedb. 
                        # use the other audio paths in the frame to generate embeddings for queries
                        all_audio_files.append(str(full_path))
                except RuntimeError as e:
                    countInvalid+=1
                    # print(f"This audio file {full_path} is not valid (probably corrupted), so nothing is happening")

    # === Step 3: Remove unmatched entries ===
    # This will only keep entries that were matched with a .wav file, basically deleted any "liked files" whose audio does not actually exist
    filename_to_metadata = {
        fname: metadata
        for fname, metadata in filename_to_metadata.items()
        if fname in matched_filenames
    }
    # print("len of all audio files is ", len(all_audio_files))
    # print("len of all metadata is ", len(filename_to_metadata))
    # print(f"There are {countInvalid} invalid files")
    # print(f"There are {countLessThanFive} audio recordings less than 5 seconds") #3464 audio recordings less than 5 seconds
    #no longer make it a key-value pair. now metadata_list is a list of dictionaries of liked sounds that are ready to be inserted into lancedb
    metadata_list = list(filename_to_metadata.values())



    #Connect to LanceDB and make schema to save the embeddings
    uri = "database/music_db.lance"
    db = lancedb.connect(uri)
    #delete the table for testing purposes, so that the same embeddings are not re-inserted because id is simply a key not a primary key, so embeddings can get reinserted
    if "music_embeddings" in db.table_names():
        #print("Table exists. If you run the next couple code blocks again, then you will get duplicate embeddings.")
        db.drop_table("music_embeddings")
    schema = pa.schema([
        pa.field("FileName", pa.string()),
        pa.field("Format", pa.string()),
        pa.field("Note", pa.string()),
        pa.field("Take", pa.string()),
        pa.field("Scene", pa.string()),
        pa.field("Project", pa.string()), 
        pa.field("Category", pa.string()), 
        pa.field("Library", pa.string()), 
        pa.field("Tape", pa.string()), 
        pa.field("Channels", pa.string()), 
        pa.field("Originator", pa.string()), 
        pa.field("Reference", pa.string()), 
        pa.field("Description", pa.string()),
        pa.field("Duration", pa.string()), #REMEMBER! duration should store the start time to end time of the embedding as a string so be sure to change that. this is because the embedding generates an array of 5 second chunks
        pa.field("FilePath", pa.string()),  
        pa.field("vector_embedding", pa.list_(pa.float32(), list_size=1280)),
    ])
    table = db.create_table("music_embeddings", schema=schema)

    # # Generate vector embeddings of all audios using perch embeddings and insert into lancedb
    model=bmz.Perch()
    # countOfLessThan5=0
    for curr_wav_file in all_audio_files:
        if (librosa.get_duration(curr_wav_file)<5):
            wav_file_looped, sample_rate = pad_short_clip(curr_wav_file)
            embedding = model.embed(wav_file_looped, sr=sample_rate)  # shape: (1, 1280)
            print("Embedding shape:", embedding.shape)
    #     #embed this batch
    #     embedding= model.embed(curr_wav_file)
    #     numChunks= embedding.shape[0]
    #     if (numChunks==0) :
    #         print("numChunks was 0 for ", str(curr_wav_file))
    #         countOfLessThan5+=1
    #     print("numChunks is ", numChunks)
    #     #insert into lancedb
    #     #ISSUE TO THINK OF: how can we batch change the Duration on line 96

        



#TODO!!
# generate vector embeddings of liked sounds using perch embeddings
# insert vector embeddings of all sounds into lancedb
# do similarity searches of liked sounds in lancedb (using perch embeddings)