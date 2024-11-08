import os
import subprocess
import traceback
from tqdm.auto import tqdm
import shutil

removed = 0
for run_dir in tqdm(os.listdir("/mount/arbeitsdaten/asr-2/vaethdk/cts_newcodebase_weights/")):
    weight_dir = f"/mount/arbeitsdaten/asr-2/vaethdk/cts_newcodebase_weights/{run_dir}/best_eval/weights"
    if os.path.exists(weight_dir):
        # find all checkpoints for run
        candidates = [cand for cand in os.listdir(weight_dir) if cand.endswith(".zip")]
        if len(candidates) == 0:
            continue
        
        # exclude last checkpoint
        last_ckpt_num = sorted([int(cand.replace("ckpt_", "").replace("_cleaned", "").replace(".zip", "")) for cand in candidates], reverse=True)[0]
        last_ckpt_name = f"ckpt_{last_ckpt_num}.zip"
        candidates.remove(last_ckpt_name)
        print(f"IGNORING LAST CHECKPOINT {last_ckpt_name}")

        for candidate in candidates:
            print("==============")
            print("CANDIDATE", candidate)
            tmp_path = f"{weight_dir}/tmp2"
            if "_cleaned" in candidate:
                print("SKIP: ALREADY CLEANED")
                continue # already cleaned

            # remove buffer
            try:
                print("Cleaning ", run_dir, ": ", candidate)
                if os.path.exists(tmp_path):
                    print("- tmp dir already exists")
                    shutil.rmtree(tmp_path)
                subprocess.call(['unzip', '-j', f"{weight_dir}/{candidate}", '-d', tmp_path])
                # delete files
                if os.path.exists(f"{tmp_path}/replay_buffer.pth"):
                    print("- remove replay buffer")
                    os.remove(f"{tmp_path}/replay_buffer.pth")
                if os.path.exists(f"{tmp_path}/policy.optimizer.pth"):
                    print("- remove optimizer")
                    os.remove(f"{tmp_path}/policy.optimizer.pth")
                # re-zip if we have weights
                if os.path.exists(f"{tmp_path}/policy.pth"):
                    if 0 == subprocess.call(['zip', '-r', f"{weight_dir}/{candidate.replace('.zip', '')}_cleaned.zip", tmp_path]):
                        print("SUCCESSFULLY PACKAGED, CLEANING ORIGINAL")
                        os.remove(f"{weight_dir}/{candidate}") # remove original checkpoints
                    shutil.rmtree(tmp_path) # remove temporary directory
                    removed += 1
                    print("REMOVED:", removed)
            except:
                traceback.print_exc()
                exit()


