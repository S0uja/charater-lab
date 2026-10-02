CHARACTER LAB V6
================

Run:
1. Make sure ComfyUI is available at C:\AI\ComfyUI.
2. Make sure sd-scripts is available at C:\AI\sd-scripts for LoRA training.
3. Install requests if needed:
   python -m pip install -r requirements.txt
4. Double-click START_Character_Lab.bat.

IMPORTANT PATH CHANGE
---------------------
All character data is now stored INSIDE this Character Lab folder:

  .\characters\Character_01\

Each character contains:
  source\             original reference images
  generated_dataset\ generated variants + matching .txt prompts
  approved\          selected training images + matching .txt prompts
  captions\          caption workspace
  config\            LoRA dataset config
  samples\           samples
  output\            trained LoRA files

Generation
----------
- Default generation count is 20.
- 20 training-oriented prompts are used; each slot has its own prompt.
- Every generated image receives a matching .txt prompt next to it.
- Each ready image can be selected and saved to the training dataset.
- "Save all ready" saves every completed image with its prompt.
- "Retry" regenerates only that slot using the same training prompt.
- During retry the old image is blurred and the slot waits for the new image.
- The new image replaces the old one only after generation finishes.

External AI paths
-----------------
The app itself and character data are local to this folder.
ComfyUI and sd-scripts remain external at C:\AI because they are shared runtimes.
