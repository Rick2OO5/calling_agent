
import os, re, json, tempfile, subprocess, wave, requests
from dotenv import load_dotenv
from flask import Flask, request, jsonify, send_from_directory
from vosk import Model, KaldiRecognizer
import pyttsx3
import google.generativeai as genai

load_dotenv()
app = Flask(__name__)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
VOSK_MODEL_PATH = os.getenv("VOSK_MODEL_PATH")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "http://localhost:5000")
AST_SOUNDS_DIR = os.getenv("AST_SOUNDS_DIR", "/usr/src/app/tts_out")
os.makedirs(AST_SOUNDS_DIR, exist_ok=True)

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-2.5-flash-lite")

def ffmpeg_to_16k(src, dst):
    cmd = ["ffmpeg","-y","-i", src, "-ar", "16000", "-ac", "1", dst]
    subprocess.run(cmd, check=True)

def transcribe_vosk(wav16):
    wf = wave.open(wav16, "rb")
    if wf.getnchannels() != 1 or wf.getframerate() != 16000:
        raise RuntimeError("Vosk needs mono 16k WAV")
    model_local = Model(VOSK_MODEL_PATH)
    rec = KaldiRecognizer(model_local, wf.getframerate())
    parts = []
    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            res = json.loads(rec.Result())
            if res.get("text"): parts.append(res.get("text"))
    final = json.loads(rec.FinalResult())
    if final.get("text"): parts.append(final.get("text"))
    return " ".join(parts).strip()

def get_gemini_structured(transcript):
    prompt = f"""You are a multilingual health assistant. Return ONLY JSON of form {{ "reply": string, "location": string|null }}.
User transcript: """{transcript}"""" 
    resp = model.generate_content(prompt, max_output_tokens=512)
    txt = ""
    if resp and resp.candidates:
        txt = resp.candidates[0].content.parts[0].text.strip()
    m = re.search(r'(\{.*\})', txt, re.DOTALL)
    jtxt = m.group(1) if m else txt
    try:
        parsed = json.loads(jtxt)
        return {"reply": parsed.get("reply",""), "location": parsed.get("location")}
    except Exception:
        return {"reply": txt or "Sorry.", "location": None}

def geocode(address):
    url = "https://maps.googleapis.com/maps/api/geocode/json"
    r = requests.get(url, params={"address": address, "key": GOOGLE_API_KEY}, timeout=10).json()
    if r.get("status") == "OK" and r.get("results"):
        loc = r["results"][0]["geometry"]["location"]
        return loc["lat"], loc["lng"]
    return None, None

def find_places(lat,lng,ptype="doctor",max_results=3):
    url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
    r = requests.get(url, params={"location":f"{lat},{lng}","radius":2000,"type":ptype,"key":GOOGLE_API_KEY}, timeout=10).json()
    out=[]
    for it in r.get("results",[])[:max_results]:
        pid = it.get("place_id")
        name = it.get("name")
        vic = it.get("vicinity")
        det = requests.get("https://maps.googleapis.com/maps/api/place/details/json",
                           params={"place_id":pid,"fields":"formatted_phone_number,formatted_address","key":GOOGLE_API_KEY}, timeout=10).json()
        phone = det.get("result",{}).get("formatted_phone_number","N/A")
        out.append({"name":name,"vicinity":vic,"phone":phone})
    return out

def tts_save_pyttsx3(text, out_wav):
    engine = pyttsx3.init()
    engine.save_to_file(text, out_wav)
    engine.runAndWait()
    return out_wav

@app.route("/process_call", methods=["POST"])
def process_call():
    uid = request.args.get("uid") or str(int(tempfile.mktemp().split('/')[-1].replace('.','')))
    # accept uploaded audio file form field 'audio'
    if "audio" in request.files:
        f = request.files["audio"]
        tmp_in = tempfile.mktemp(suffix=os.path.splitext(f.filename)[1])
        f.save(tmp_in)
    else:
        # try to accept 'audio' field path or file argument name 'file'
        return jsonify({"error":"no audio uploaded"}), 400

    try:
        tmp16 = tempfile.mktemp(suffix=".wav")
        ffmpeg_to_16k(tmp_in, tmp16)
        transcript = transcribe_vosk(tmp16)
        gem = get_gemini_structured(transcript)
        reply = gem.get("reply","")
        location = gem.get("location")
        doctors=[]; pharms=[]
        if location:
            lat,lng = geocode(location)
            if lat:
                doctors = find_places(lat,lng,"doctor",3)
                pharms = find_places(lat,lng,"pharmacy",3)
        parts=[reply]
        if doctors:
            parts.append("Nearby doctors:")
            for d in doctors:
                parts.append(f"{d['name']}, {d['vicinity']}. Phone: {d['phone']}.")
        if pharms:
            parts.append("Nearby pharmacies:")
            for p in pharms:
                parts.append(f"{p['name']}, {p['vicinity']}. Phone: {p['phone']}.")
        parts.append("If it is serious, please visit a hospital immediately.")
        final = " ".join(parts)
        # TTS
        out_wav = os.path.join(AST_SOUNDS_DIR, f"response_{uid}.wav")
        tts_save_pyttsx3(final, out_wav)
        # return path info
        public = f"{PUBLIC_BASE_URL}/tts/response_{uid}.wav"
        return jsonify({"status":"ok","transcript":transcript,"reply":final,"tts_url":public}), 200
    except Exception as e:
        return jsonify({"error":str(e)}),500

@app.route("/tts/<path:filename>")
def serve_tts(filename):
    return send_from_directory(AST_SOUNDS_DIR, filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
