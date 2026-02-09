![Piper](etc/logo.png)

A fast and local neural text-to-speech engine that embeds [espeak-ng][] for phonemization.

Install with:

``` sh
pip install piper-tts
```

* 🎧 [Samples][samples]
* 💡 [Demo][demo]
* 🗣️ [Voices][voices]
* 🖥️ [Command-line interface][cli]
* 🌐 [Web server][api-http]
* 🐍 [Python API][api-python]
* 🔧 [C/C++ API][libpiper]
* 🏋️ [Training new voices][training]
* 🛠️ [Building manually][building]

---

## Internal Voice Training Workflow

Vietnamese fientune procedure 
### 1. Dataset Preparation

Use [piper-recording-studio](https://github.com/rhasspy/piper-recording-studio) to record and prepare raw data.

* 
**Environment Setup**: 



```bash
cd ~/
git clone https://github.com/rhasspy/piper-recording-studio.git
python3 -m venv ~/piper-recording-studio/.venv
cd ~/piper-recording-studio/
[cite_start]source ~/piper-recording-studio/.venv/bin/activate [cite: 4, 5, 6, 7, 8]
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
[cite_start]python3 -m pip install -r requirements_export.txt [cite: 9, 10, 11]

```

* **Procedure**:
* Launch recording tool: `python3 -m piper_recording_studio` 


* A new file dataset bonus for Vietnamese: [5000000001_5000000450_TNhi.txt](https://github.com/nuverxai/piper1-gpl-vn/tree/feat/hybrid/extra-recording-file/5000000001_5000000450_TNhi.txt) 
    * Put file in to the Vietnamese prompt folder: [vi-VN](https://github.com/rhasspy/piper-recording-studio/tree/master/prompts/Vietnamese%20(Vietnam)_vi-VN)


* Export dataset: `python3 -m export_dataset output/vi-VN ~/piper-recording-studio/dataset-name` 




* 
**Technical Optimization**: 


* Check and modify the main file to set the Resample rate to 22050Hz. Code update (Check and update) [Code](https://github.com/nuverxai/piper1-gpl-vn/tree/feat/hybrid/extra-recording-file/__main__.py)


* If audio issues occur (e.g., clipped sound), adjust export parameters: `threshold: 0.3` and `keep-chunks: 10`.




### 2. Training on Google Colab


* 
**Setup Colab**: Sử dụng cấu hình tại [nuverxai/piper1-gpl-vn](https://github.com/nuverxai/piper1-gpl-vn/tree/feat/hybrid/colab/VN_piper_demo.ipynb). 


* 
**Input Data**: Compress the dataset folder, upload to Google Drive, and set permissions to Public. -> Get the file ID to change in Colab 



* 
**Checkpoint Selection**: 


* Vietnamese: [vais1000/medium](https://huggingface.co/datasets/rhasspy/piper-checkpoints/tree/main/vi/vi_VN/vais1000/medium) 


* English: [Male](https://huggingface.co/datasets/rhasspy/piper-checkpoints/tree/main/en/en_US/hfc_male/medium) | [Female](https://huggingface.co/datasets/rhasspy/piper-checkpoints/tree/main/en/en_US/hfc_female/medium) 




* 
**Epoch Configuration**: Set --trainer.max_epochs to the sum of previous epochs plus the new training target.

---


People/projects using Piper:

* [Home Assistant](https://github.com/home-assistant/addons/blob/master/piper/README.md)
* [NVDA - NonVisual Desktop Access](https://www.nvaccess.org/post/in-process-8th-may-2023/#voices)
* [Image Captioning for the Visually Impaired and Blind: A Recipe for Low-Resource Languages](https://www.techrxiv.org/articles/preprint/Image_Captioning_for_the_Visually_Impaired_and_Blind_A_Recipe_for_Low-Resource_Languages/22133894)
* [Video tutorial by Thorsten Müller](https://youtu.be/rjq5eZoWWSo)
* [Open Voice Operating System](https://github.com/OpenVoiceOS/ovos-tts-plugin-piper)
* [JetsonGPT](https://github.com/shahizat/jetsonGPT)
* [LocalAI](https://github.com/go-skynet/LocalAI)
* [Lernstick EDU / EXAM: reading clipboard content aloud with language detection](https://lernstick.ch/)
* [Natural Speech - A plugin for Runelite, an OSRS Client](https://github.com/phyce/rl-natural-speech)
* [mintPiper](https://github.com/evuraan/mintPiper)
* [Vim-Piper](https://github.com/wolandark/vim-piper)
* [POTaTOS](https://www.youtube.com/watch?v=Dz95q6XYjwY)
* [Narration Studio](https://github.com/phyce/Narration-Studio)
* [Basic TTS](https://basictts.com/) - Simple online text-to-speech converter.

[![A library from the Open Home Foundation](https://www.openhomefoundation.org/badges/ohf-library.png)](https://www.openhomefoundation.org/)

<!-- Links -->
[espeak-ng]: https://github.com/espeak-ng/espeak-ng
[cli]: https://github.com/OHF-Voice/piper1-gpl/blob/main/docs/CLI.md
[api-http]: https://github.com/OHF-Voice/piper1-gpl/blob/main/docs/API_HTTP.md
[api-python]: https://github.com/OHF-Voice/piper1-gpl/blob/main/docs/API_PYTHON.md
[training]: https://github.com/OHF-Voice/piper1-gpl/blob/main/docs/TRAINING.md
[building]: https://github.com/OHF-Voice/piper1-gpl/blob/main/docs/BUILDING.md
[voices]: https://github.com/OHF-Voice/piper1-gpl/blob/main/docs/VOICES.md
[samples]: https://rhasspy.github.io/piper-samples
[demo]: https://rhasspy.github.io/piper-samples/demo.html
[libpiper]: https://github.com/OHF-Voice/piper1-gpl/tree/main/libpiper
