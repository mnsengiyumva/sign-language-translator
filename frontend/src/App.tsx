import { useState, useEffect, useRef, useCallback } from "react";

/*
  App.tsx — Main React component for the Sign Language Translator
  
  WHAT THIS DOES:
    - Opens the user's webcam using the browser's getUserMedia API
    - Sends frames to the FastAPI backend every 100ms via WebSocket
    - Receives predicted ASL letters back and displays them
    - Builds words and sentences from recognized letters
    - Speaks the sentence aloud using the Web Speech API
*/

// How often to send a frame to the backend (milliseconds)
// 100ms = ~10 frames per second, fast enough for smooth recognition
const FRAME_INTERVAL_MS = 100;

// Minimum confidence % to display a prediction
const CONFIDENCE_THRESHOLD = 85;

// How many times we must see the same letter before adding it to the word
// Prevents accidental letter additions from brief sign changes
const LETTER_CONFIRM_FRAMES = 8;

export default function App() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const frameIntervalRef = useRef<number | null>(null);
  const letterCountRef = useRef<{ letter: string; count: number }>({ letter: "", count: 0 });

  const [connected, setConnected] = useState(false);
  const [currentLetter, setCurrentLetter] = useState<string>("");
  const [confidence, setConfidence] = useState<number>(0);
  const [handDetected, setHandDetected] = useState(false);
  const [currentWord, setCurrentWord] = useState("");
  const [sentence, setSentence] = useState<string[]>([]);
  const [status, setStatus] = useState("Connecting to server...");
  const [letterProgress, setLetterProgress] = useState(0);

  // ==========================================================================
  // CONNECT TO WEBSOCKET SERVER
  // ==========================================================================

  useEffect(() => {
    const ws = new WebSocket("ws://localhost:8000/ws");
    wsRef.current = ws;

    ws.onopen = () => {
      setConnected(true);
      setStatus("Connected — show your hand to the camera");
    };

    ws.onmessage = (event) => {
      // Parse the prediction sent back from the backend
      const data = JSON.parse(event.data);
      setHandDetected(data.hand_detected);

      if (data.hand_detected && data.confidence >= CONFIDENCE_THRESHOLD) {
        setCurrentLetter(data.letter);
        setConfidence(data.confidence);

        // Track how many consecutive frames show the same letter
        // Only add to word once we've seen it LETTER_CONFIRM_FRAMES times
        if (data.letter === letterCountRef.current.letter) {
          letterCountRef.current.count += 1;
          setLetterProgress(Math.min(letterCountRef.current.count / LETTER_CONFIRM_FRAMES, 1));

          if (letterCountRef.current.count === LETTER_CONFIRM_FRAMES) {
            if (data.letter !== "nothing") {
              setCurrentWord((prev) => prev + data.letter);
            }
            letterCountRef.current = { letter: "", count: 0 };
            setLetterProgress(0);
          }
        } else {
          letterCountRef.current = { letter: data.letter, count: 1 };
          setLetterProgress(0);
        }
      } else {
        setCurrentLetter("");
        setConfidence(0);
        letterCountRef.current = { letter: "", count: 0 };
        setLetterProgress(0);
      }
    };

    ws.onclose = () => {
      setConnected(false);
      setStatus("Disconnected from server");
    };

    ws.onerror = () => {
      setStatus("Cannot connect to server — is it running on port 8000?");
    };

    return () => ws.close();
  }, []);

  // ==========================================================================
  // START WEBCAM
  // ==========================================================================

  useEffect(() => {
    async function startCamera() {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { width: 640, height: 480, facingMode: "user" }
        });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
      } catch {
        setStatus("Camera access denied — please allow camera in browser settings");
      }
    }
    startCamera();
  }, []);

  // ==========================================================================
  // SEND FRAMES TO BACKEND
  // ==========================================================================

  const sendFrame = useCallback(() => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    const ws = wsRef.current;

    if (!video || !canvas || !ws || ws.readyState !== WebSocket.OPEN) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Draw the current video frame onto the hidden canvas
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    // Convert canvas to base64 JPEG and send to backend
    const frameData = canvas.toDataURL("image/jpeg", 0.8);
    ws.send(frameData);
  }, []);

  useEffect(() => {
    if (connected) {
      frameIntervalRef.current = window.setInterval(sendFrame, FRAME_INTERVAL_MS);
    }
    return () => {
      if (frameIntervalRef.current) clearInterval(frameIntervalRef.current);
    };
  }, [connected, sendFrame]);

  // ==========================================================================
  // WORD AND SENTENCE ACTIONS
  // ==========================================================================

  function addWord() {
    if (currentWord.trim()) {
      setSentence((prev) => [...prev, currentWord]);
      setCurrentWord("");
    }
  }

  function deleteLetter() {
    setCurrentWord((prev) => prev.slice(0, -1));
  }

  function clearAll() {
    setCurrentWord("");
    setSentence([]);
  }

  function speakSentence() {
    // Use the browser's built-in Web Speech API to speak the sentence aloud
    const fullText = [...sentence, currentWord].join(" ").trim();
    if (!fullText) return;
    const utterance = new SpeechSynthesisUtterance(fullText);
    utterance.rate = 0.9;
    utterance.pitch = 1;
    window.speechSynthesis.speak(utterance);
  }

  const fullSentence = [...sentence, currentWord].join(" ").trim();

  // ==========================================================================
  // RENDER
  // ==========================================================================

  return (
    <div style={styles.app}>
      {/* ---- HEADER ---- */}
      <header style={styles.header}>
        <h1 style={styles.title}>�� Sign Language Translator</h1>
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <div style={{
            ...styles.statusDot,
            backgroundColor: connected ? "#00e5ff" : "#ff5252"
          }} />
          <span style={styles.statusText}>{status}</span>
        </div>
      </header>

      {/* ---- MAIN CONTENT ---- */}
      <main style={styles.main}>

        {/* ---- LEFT: CAMERA FEED ---- */}
        <section style={styles.cameraSection}>
          <h2 style={styles.sectionTitle}>Camera Feed</h2>
          <div style={styles.videoWrapper}>
            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              style={styles.video}
            />
            {/* Hidden canvas used to capture frames to send to backend */}
            <canvas
              ref={canvasRef}
              width={640}
              height={480}
              style={{ display: "none" }}
            />
            {/* Hand detection indicator overlay */}
            <div style={{
              ...styles.handBadge,
              backgroundColor: handDetected ? "#00e5ff22" : "#ffffff11",
              borderColor: handDetected ? "#00e5ff" : "#ffffff33"
            }}>
              {handDetected ? "✋ Hand Detected" : "No Hand"}
            </div>
          </div>
        </section>

        {/* ---- RIGHT: OUTPUT PANEL ---- */}
        <section style={styles.outputSection}>

          {/* Current predicted letter */}
          <div style={styles.letterCard}>
            <p style={styles.cardLabel}>Detected Sign</p>
            <div style={styles.bigLetter}>
              {currentLetter && currentLetter !== "nothing"
                ? currentLetter.toUpperCase()
                : "—"}
            </div>
            {confidence > 0 && (
              <p style={styles.confidence}>{confidence}% confidence</p>
            )}
            {/* Progress bar — fills as you hold the letter steady */}
            <div style={styles.progressBarBg}>
              <div style={{
                ...styles.progressBarFill,
                width: `${letterProgress * 100}%`
              }} />
            </div>
            <p style={styles.progressLabel}>
              {letterProgress > 0 ? "Hold steady..." : "Show a sign"}
            </p>
          </div>

          {/* Current word being built */}
          <div style={styles.wordCard}>
            <p style={styles.cardLabel}>Current Word</p>
            <p style={styles.wordText}>
              {currentWord || <span style={{ opacity: 0.3 }}>Start signing...</span>}
            </p>
            <div style={styles.wordActions}>
              <button style={styles.btnSecondary} onClick={deleteLetter}>
                ⌫ Delete
              </button>
              <button style={styles.btnPrimary} onClick={addWord}>
                Add Word →
              </button>
            </div>
          </div>

          {/* Full sentence */}
          <div style={styles.sentenceCard}>
            <p style={styles.cardLabel}>Sentence</p>
            <p style={styles.sentenceText}>
              {fullSentence || <span style={{ opacity: 0.3 }}>Your sentence will appear here...</span>}
            </p>
            <div style={styles.sentenceActions}>
              <button style={styles.btnDanger} onClick={clearAll}>
                🗑 Clear
              </button>
              <button style={styles.btnSpeak} onClick={speakSentence}>
                🔊 Speak
              </button>
            </div>
          </div>

        </section>
      </main>
    </div>
  );
}

// =============================================================================
// STYLES
// =============================================================================

const styles: { [key: string]: React.CSSProperties } = {
  app: {
    minHeight: "100vh",
    backgroundColor: "#0a0f1e",
    color: "#ffffff",
    fontFamily: "'Segoe UI', sans-serif",
    display: "flex",
    flexDirection: "column",
  },
  header: {
    backgroundColor: "#0d1530",
    borderBottom: "1px solid #1e2d5a",
    padding: "16px 32px",
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
  },
  title: {
    margin: 0,
    fontSize: 22,
    fontWeight: 700,
    color: "#ffffff",
    letterSpacing: 0.5,
  },
  statusDot: {
    width: 10,
    height: 10,
    borderRadius: "50%",
  },
  statusText: {
    fontSize: 13,
    color: "#a0aec0",
  },
  main: {
    display: "flex",
    flex: 1,
    gap: 24,
    padding: 24,
  },
  cameraSection: {
    flex: 1.2,
    display: "flex",
    flexDirection: "column",
    gap: 12,
  },
  outputSection: {
    flex: 1,
    display: "flex",
    flexDirection: "column",
    gap: 16,
  },
  sectionTitle: {
    margin: 0,
    fontSize: 14,
    fontWeight: 600,
    color: "#a0aec0",
    textTransform: "uppercase",
    letterSpacing: 1,
  },
  videoWrapper: {
    position: "relative",
    borderRadius: 16,
    overflow: "hidden",
    border: "1px solid #1e2d5a",
    backgroundColor: "#060b18",
  },
  video: {
    width: "100%",
    display: "block",
    transform: "scaleX(-1)", // mirror the feed
    borderRadius: 16,
  },
  handBadge: {
    position: "absolute",
    top: 12,
    left: 12,
    padding: "6px 14px",
    borderRadius: 20,
    border: "1px solid",
    fontSize: 13,
    fontWeight: 600,
    backdropFilter: "blur(8px)",
    color: "#ffffff",
  },
  letterCard: {
    backgroundColor: "#0d1530",
    border: "1px solid #1e2d5a",
    borderRadius: 16,
    padding: 20,
    textAlign: "center",
  },
  cardLabel: {
    margin: "0 0 8px",
    fontSize: 12,
    fontWeight: 600,
    color: "#a0aec0",
    textTransform: "uppercase",
    letterSpacing: 1,
  },
  bigLetter: {
    fontSize: 80,
    fontWeight: 800,
    color: "#00e5ff",
    lineHeight: 1,
    margin: "8px 0",
  },
  confidence: {
    margin: "4px 0 12px",
    fontSize: 13,
    color: "#a0aec0",
  },
  progressBarBg: {
    height: 6,
    backgroundColor: "#1e2d5a",
    borderRadius: 3,
    overflow: "hidden",
    margin: "8px 0 6px",
  },
  progressBarFill: {
    height: "100%",
    backgroundColor: "#00e5ff",
    borderRadius: 3,
    transition: "width 0.1s ease",
  },
  progressLabel: {
    margin: 0,
    fontSize: 12,
    color: "#4a5568",
  },
  wordCard: {
    backgroundColor: "#0d1530",
    border: "1px solid #1e2d5a",
    borderRadius: 16,
    padding: 20,
  },
  wordText: {
    fontSize: 28,
    fontWeight: 700,
    color: "#ffffff",
    margin: "8px 0 16px",
    minHeight: 40,
    letterSpacing: 2,
  },
  wordActions: {
    display: "flex",
    gap: 10,
  },
  sentenceCard: {
    backgroundColor: "#0d1530",
    border: "1px solid #1e2d5a",
    borderRadius: 16,
    padding: 20,
    flex: 1,
  },
  sentenceText: {
    fontSize: 18,
    color: "#e2e8f0",
    margin: "8px 0 16px",
    lineHeight: 1.6,
    minHeight: 60,
  },
  sentenceActions: {
    display: "flex",
    gap: 10,
  },
  btnPrimary: {
    backgroundColor: "#00e5ff",
    color: "#0a0f1e",
    border: "none",
    borderRadius: 10,
    padding: "10px 20px",
    fontWeight: 700,
    fontSize: 14,
    cursor: "pointer",
    flex: 1,
  },
  btnSecondary: {
    backgroundColor: "#1e2d5a",
    color: "#ffffff",
    border: "1px solid #2d3f7a",
    borderRadius: 10,
    padding: "10px 20px",
    fontWeight: 600,
    fontSize: 14,
    cursor: "pointer",
  },
  btnDanger: {
    backgroundColor: "#1e2d5a",
    color: "#ff5252",
    border: "1px solid #ff525233",
    borderRadius: 10,
    padding: "10px 20px",
    fontWeight: 600,
    fontSize: 14,
    cursor: "pointer",
  },
  btnSpeak: {
    backgroundColor: "#00e5ff",
    color: "#0a0f1e",
    border: "none",
    borderRadius: 10,
    padding: "10px 20px",
    fontWeight: 700,
    fontSize: 14,
    cursor: "pointer",
    flex: 1,
  },
};
