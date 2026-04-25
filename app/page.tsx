"use client";

import { useEffect, useMemo, useRef, useState } from "react";

type PredictResult = { label: string; confidence: number } | null;

const HOLD_FRAMES_NEEDED = 25;
const CONFIDENCE_THRESH = 0.4;

function clamp01(x: number) {
  return Math.max(0, Math.min(1, x));
}

function softmax(logits: Float32Array) {
  let max = -Infinity;
  for (let i = 0; i < logits.length; i++) max = Math.max(max, logits[i]!);
  let sum = 0;
  const exps = new Float32Array(logits.length);
  for (let i = 0; i < logits.length; i++) {
    const v = Math.exp(logits[i]! - max);
    exps[i] = v;
    sum += v;
  }
  for (let i = 0; i < exps.length; i++) exps[i] = exps[i]! / (sum || 1);
  return exps;
}

export default function Page() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  const [ready, setReady] = useState(false);
  const [status, setStatus] = useState<string>("Loading models…");
  const [text, setText] = useState("");
  const [letter, setLetter] = useState<string>("");
  const [confidence, setConfidence] = useState<number>(0);
  const [holdCount, setHoldCount] = useState<number>(0);

  const [classes, setClasses] = useState<string[] | null>(null);

  const sessionRef = useRef({
    currentLetter: "",
    holdCount: 0,
    text: "",
  });

  const uiText = useMemo(() => text, [text]);

  useEffect(() => {
    let stop = false;

    async function init() {
      try {
        setStatus("Requesting webcam permission…");
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { width: 640, height: 480 },
          audio: false,
        });
        if (stop) return;
        const video = videoRef.current;
        if (!video) return;
        video.srcObject = stream;
        await video.play();

        setStatus("Loading hand landmarker + ONNX model…");

        // Lazy imports (client-only).
        const vision = await import("@mediapipe/tasks-vision");
        const ort = await import("onnxruntime-web");

        // Load MediaPipe hand landmarker.
        const fileset = await vision.FilesetResolver.forVisionTasks(
          "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.21/wasm"
        );
        const handLandmarker = await vision.HandLandmarker.createFromOptions(
          fileset,
          {
            baseOptions: {
              modelAssetPath:
                "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
            },
            numHands: 1,
            minHandDetectionConfidence: 0.25,
            minHandPresenceConfidence: 0.25,
            minTrackingConfidence: 0.25,
          }
        );

        // Load labels + ONNX model (exported from your sklearn pipeline).
        const classesRes = await fetch("/classes.json");
        if (!classesRes.ok) throw new Error("Missing /classes.json");
        const cls = (await classesRes.json()) as { classes: string[] };
        setClasses(cls.classes);

        const session = await ort.InferenceSession.create("/model.onnx", {
          executionProviders: ["wasm"],
        });

        setStatus("Running…");
        setReady(true);

        const canvas = canvasRef.current;
        const ctx = canvas?.getContext("2d") ?? null;

        const runLoop = async () => {
          if (stop) return;
          const v = videoRef.current;
          const c = canvasRef.current;
          if (!v || !c) return;

          const w = v.videoWidth || 640;
          const h = v.videoHeight || 480;
          if (c.width !== w) c.width = w;
          if (c.height !== h) c.height = h;

          // Draw preview.
          if (ctx) {
            ctx.drawImage(v, 0, 0, w, h);
          }

          const now = performance.now();
          const result = handLandmarker.detectForVideo(v, now);

          // Mirror handling: Python code tries normal first, then mirrored.
          // In browser, we just use the raw output and normalize by wrist,
          // which tends to be robust; if you need perfect parity, we can
          // add a second pass with a mirrored draw.
          const pred = await predictFromResult(result, session, cls.classes, ort);

          if (ctx) {
            drawOverlay(ctx, result);
          }

          updateHoldAndText(pred);

          requestAnimationFrame(runLoop);
        };

        const updateHoldAndText = (pred: PredictResult) => {
          const s = sessionRef.current;
          const nextLetter = pred?.label ?? "";
          const nextConf = pred?.confidence ?? 0;

          let acceptedLetter = "";
          if (nextLetter && nextConf >= CONFIDENCE_THRESH) {
            acceptedLetter = nextLetter;
          }

          if (acceptedLetter && acceptedLetter === s.currentLetter) {
            s.holdCount += 1;
          } else {
            s.currentLetter = acceptedLetter;
            s.holdCount = 0;
          }

          if (s.holdCount >= HOLD_FRAMES_NEEDED) {
            if (acceptedLetter && acceptedLetter.toLowerCase() !== "nothing") {
              if (acceptedLetter.toLowerCase() === "space") s.text += " ";
              else if (acceptedLetter.toLowerCase() === "del")
                s.text = s.text.slice(0, -1);
              else s.text += acceptedLetter;
            }
            s.holdCount = 0;
          }

          setLetter(acceptedLetter);
          setConfidence(nextConf);
          setHoldCount(s.holdCount);
          setText(s.text);
        };

        requestAnimationFrame(runLoop);
      } catch (e: any) {
        setStatus(e?.message ?? String(e));
      }
    }

    init();
    return () => {
      stop = true;
      const v = videoRef.current;
      const stream = v?.srcObject as MediaStream | null;
      stream?.getTracks().forEach((t) => t.stop());
    };
  }, []);

  return (
    <main
      style={{
        minHeight: "100vh",
        display: "flex",
        flexDirection: "column",
        gap: 16,
        padding: 20,
        boxSizing: "border-box",
        maxWidth: 1100,
        margin: "0 auto",
      }}
    >
      <header style={{ display: "flex", justifyContent: "space-between", gap: 12 }}>
        <div>
          <div style={{ fontSize: 20, fontWeight: 700 }}>ASL Detector (Vercel-ready)</div>
          <div style={{ opacity: 0.8, marginTop: 4, fontSize: 13 }}>{status}</div>
        </div>
        <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
          <Badge label={ready ? "LIVE" : "INIT"} tone={ready ? "ok" : "warn"} />
          <Badge
            label={classes ? `${classes.length} classes` : "labels missing"}
            tone={classes ? "ok" : "bad"}
          />
        </div>
      </header>

      <section
        style={{
          display: "grid",
          gridTemplateColumns: "2fr 1fr",
          gap: 16,
          alignItems: "start",
        }}
      >
        <div
          style={{
            position: "relative",
            borderRadius: 14,
            overflow: "hidden",
            border: "1px solid rgba(255,255,255,0.12)",
            background: "rgba(255,255,255,0.03)",
          }}
        >
          <video
            ref={videoRef}
            playsInline
            muted
            style={{ display: "none" }}
          />
          <canvas ref={canvasRef} style={{ width: "100%", height: "auto" }} />
        </div>

        <div
          style={{
            display: "flex",
            flexDirection: "column",
            gap: 12,
            padding: 14,
            borderRadius: 14,
            border: "1px solid rgba(255,255,255,0.12)",
            background: "rgba(255,255,255,0.03)",
          }}
        >
          <div style={{ display: "flex", justifyContent: "space-between", gap: 12 }}>
            <div>
              <div style={{ opacity: 0.8, fontSize: 12 }}>Detected</div>
              <div style={{ fontSize: 40, fontWeight: 800, lineHeight: 1 }}>
                {letter || "?"}
              </div>
            </div>
            <div style={{ textAlign: "right" }}>
              <div style={{ opacity: 0.8, fontSize: 12 }}>Confidence</div>
              <div style={{ fontSize: 22, fontWeight: 700 }}>
                {Math.round(clamp01(confidence) * 100)}%
              </div>
            </div>
          </div>

          <div>
            <div style={{ opacity: 0.8, fontSize: 12, marginBottom: 6 }}>
              Hold to type ({holdCount}/{HOLD_FRAMES_NEEDED})
            </div>
            <div
              style={{
                height: 10,
                borderRadius: 999,
                background: "rgba(255,255,255,0.10)",
                overflow: "hidden",
              }}
            >
              <div
                style={{
                  height: "100%",
                  width: `${(holdCount / HOLD_FRAMES_NEEDED) * 100}%`,
                  background: "linear-gradient(90deg,#7c3aed,#06b6d4)",
                }}
              />
            </div>
          </div>

          <div>
            <div style={{ opacity: 0.8, fontSize: 12, marginBottom: 6 }}>Text</div>
            <div
              style={{
                minHeight: 90,
                borderRadius: 12,
                padding: 10,
                background: "rgba(0,0,0,0.35)",
                border: "1px solid rgba(255,255,255,0.10)",
                whiteSpace: "pre-wrap",
                wordBreak: "break-word",
              }}
            >
              {uiText || "—"}
            </div>
          </div>

          <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
            <Button
              onClick={() => {
                sessionRef.current.text += " ";
                setText(sessionRef.current.text);
              }}
            >
              Space
            </Button>
            <Button
              onClick={() => {
                sessionRef.current.text = sessionRef.current.text.slice(0, -1);
                setText(sessionRef.current.text);
              }}
            >
              Backspace
            </Button>
            <Button
              onClick={() => {
                sessionRef.current.text = "";
                setText("");
              }}
            >
              Clear
            </Button>
          </div>

          <div style={{ opacity: 0.7, fontSize: 12, lineHeight: 1.35 }}>
            This runs fully in the browser (webcam + hand landmarks + ONNX model),
            so it can be deployed to Vercel.
          </div>
        </div>
      </section>
    </main>
  );
}

function Badge({ label, tone }: { label: string; tone: "ok" | "warn" | "bad" }) {
  const bg =
    tone === "ok"
      ? "rgba(16,185,129,0.18)"
      : tone === "warn"
        ? "rgba(245,158,11,0.18)"
        : "rgba(239,68,68,0.18)";
  const border =
    tone === "ok"
      ? "rgba(16,185,129,0.35)"
      : tone === "warn"
        ? "rgba(245,158,11,0.35)"
        : "rgba(239,68,68,0.35)";
  return (
    <span
      style={{
        padding: "6px 10px",
        borderRadius: 999,
        background: bg,
        border: `1px solid ${border}`,
        fontSize: 12,
        fontWeight: 700,
        letterSpacing: 0.2,
      }}
    >
      {label}
    </span>
  );
}

function Button({
  children,
  onClick,
}: {
  children: React.ReactNode;
  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      style={{
        padding: "10px 12px",
        borderRadius: 12,
        border: "1px solid rgba(255,255,255,0.14)",
        background: "rgba(255,255,255,0.06)",
        color: "#e8eaf2",
        fontWeight: 700,
        cursor: "pointer",
      }}
    >
      {children}
    </button>
  );
}

function drawOverlay(ctx: CanvasRenderingContext2D, result: any) {
  const landmarks = result?.landmarks?.[0];
  if (!landmarks) return;

  ctx.save();
  ctx.lineWidth = 2;
  ctx.strokeStyle = "rgba(0,255,128,0.9)";
  ctx.fillStyle = "rgba(255,64,128,0.9)";

  // Simple skeleton based on MediaPipe hand connections.
  const connections: Array<[number, number]> = [
    [0, 1],
    [1, 2],
    [2, 3],
    [3, 4],
    [0, 5],
    [5, 6],
    [6, 7],
    [7, 8],
    [5, 9],
    [9, 10],
    [10, 11],
    [11, 12],
    [9, 13],
    [13, 14],
    [14, 15],
    [15, 16],
    [13, 17],
    [17, 18],
    [18, 19],
    [19, 20],
    [0, 17],
  ];

  const w = ctx.canvas.width;
  const h = ctx.canvas.height;

  for (const [a, b] of connections) {
    const A = landmarks[a];
    const B = landmarks[b];
    if (!A || !B) continue;
    ctx.beginPath();
    ctx.moveTo(A.x * w, A.y * h);
    ctx.lineTo(B.x * w, B.y * h);
    ctx.stroke();
  }

  for (const lm of landmarks) {
    ctx.beginPath();
    ctx.arc(lm.x * w, lm.y * h, 4, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.restore();
}

async function predictFromResult(
  result: any,
  session: any,
  classes: string[],
  ort: any
): Promise<PredictResult> {
  const hand = result?.landmarks?.[0] as Array<{ x: number; y: number; z: number }> | undefined;
  if (!hand || hand.length !== 21) return null;

  const wrist = hand[0]!;
  const vec = new Float32Array(63);
  for (let i = 0; i < 21; i++) {
    const lm = hand[i]!;
    vec[i * 3 + 0] = lm.x - wrist.x;
    vec[i * 3 + 1] = lm.y - wrist.y;
    vec[i * 3 + 2] = (lm.z ?? 0) - (wrist.z ?? 0);
  }
  let maxAbs = 0;
  for (let i = 0; i < vec.length; i++) maxAbs = Math.max(maxAbs, Math.abs(vec[i]!));
  const denom = maxAbs + 1e-6;
  for (let i = 0; i < vec.length; i++) vec[i] = vec[i]! / denom;

  const inputName = session.inputNames[0] as string;
  const feeds: Record<string, any> = {};
  feeds[inputName] = new ort.Tensor("float32", vec, [1, 63]);

  const out = await session.run(feeds);
  const outName = session.outputNames[0] as string;
  const first = out[outName];

  // Depending on export, this might be probabilities or scores.
  const data = first?.data as Float32Array | number[] | undefined;
  if (!data || data.length !== classes.length) return null;

  const arr = data instanceof Float32Array ? data : Float32Array.from(data);
  let probs: Float32Array;
  let sum = 0;
  let inRange01 = true;
  for (let i = 0; i < arr.length; i++) {
    const v = arr[i]!;
    sum += v;
    if (v < 0 || v > 1) inRange01 = false;
  }
  if (inRange01 && sum > 0.98 && sum < 1.02) probs = arr;
  else probs = softmax(arr);
  let bestI = 0;
  for (let i = 1; i < probs.length; i++) if (probs[i]! > probs[bestI]!) bestI = i;

  return { label: String(classes[bestI] ?? ""), confidence: probs[bestI]! };
}

