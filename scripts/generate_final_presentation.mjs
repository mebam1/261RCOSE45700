import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { Presentation, PresentationFile } from "@oai/artifact-tool";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const repoRoot = path.resolve(__dirname, "..");

const FONT_FAMILY = "Pretendard";
const COLORS = {
  cream: "#f2f0eb",
  ceramic: "#edebe9",
  deepGreen: "#1E3932",
  brandGreen: "#006241",
  accentGreen: "#00754A",
  mint: "#d4e9e2",
  gold: "#cba258",
  textStrong: "#1f1f1f",
  textSoft: "#5b5b57",
  white: "#ffffff",
  line: "#d7d2ca",
  softCard: "#fbfaf7",
};

const slideSize = { width: 1280, height: 720 };
const frame = { left: 72, top: 58, width: 1136, height: 604 };
const defaultFinalPptx = path.join(repoRoot, "outputs", "franchise_cleanliness_final_presentation.pptx");
const defaultWorkRoot = path.join(
  os.tmpdir(),
  "codex-presentations",
  "manual-20260618",
  "franchise-cleanliness-final",
);

const sourceImagePaths = {
  reference: path.join(repoRoot, "vlm_classifier", "frames", "frame_0000.jpg"),
  crop: path.join(repoRoot, "vlm_classifier", "table_crops", "T01_0100.jpg"),
  frameA: path.join(repoRoot, "vlm_classifier", "frames", "frame_0000.jpg"),
  frameB: path.join(repoRoot, "vlm_classifier", "frames", "frame_0080.jpg"),
  frameC: path.join(repoRoot, "vlm_classifier", "frames", "frame_0170.jpg"),
};

const slidePlanText = `Deck mode: create
Audience: 교수, 조교, 수강생
Slide count: 12
Visual direction:
- Warm cream canvas with deep green section bands
- Pretendard only
- Light text density, speaker-friendly bullets
- Rounded cards, restrained shadows, no logo use
Palette:
- cream #f2f0eb
- ceramic #edebe9
- deep green #1E3932
- heading green #006241
- accent green #00754A
- gold #cba258
Typography:
- Title: Pretendard 34-40px bold
- Section label: Pretendard 12-13px bold
- Body bullet: Pretendard 21-24px
- Caption: Pretendard 12-14px
Slide flow:
1. Title
2. Background and problem
3. Limits of supervisor visits
4. Core idea
5. System architecture
6. Vision-based table cleanliness
7. Action-based support evaluation
8. Video timing extraction and efficiency
9. Implementation structure and stack
10. Implemented results and demo flow
11. Expected effect
12. Limitations and future direction + conclusion`;

const sourceNotesText = `User-provided project topic and presentation intent
- Conversation prompt in current thread
- Supports deck narrative, audience, slide order, tone, and language constraints

Local project source: INTEGRATED_PROJECT_DEMO_REPORT.md
- Path: ${path.join(repoRoot, "INTEGRATED_PROJECT_DEMO_REPORT.md")}
- Accessed: 2026-06-18
- Supports implemented features, demo flow, main screens, APIs, workflow structure, current strengths/limits

Local project source: ACTION_WORKFLOW_API.md
- Path: ${path.join(repoRoot, "ACTION_WORKFLOW_API.md")}
- Accessed: 2026-06-18
- Supports workflow states, example response structure, final score fields, reason-code based explanation

Local project source: app/main.py
- Path: ${path.join(repoRoot, "app", "main.py")}
- Accessed: 2026-06-18
- Supports actual integrated routes, workflow-from-images, workflow-from-video, dynamic sampling, reports page

Local project assets:
- ${sourceImagePaths.reference}
- ${sourceImagePaths.crop}
- ${sourceImagePaths.frameA}
- ${sourceImagePaths.frameB}
- ${sourceImagePaths.frameC}
- Used as illustrative local sample images from the implemented repository

Design direction source:
- Starbucks-inspired design guide from user-provided attachment
- Used only for mood, palette logic, spacing rhythm, and warmth
- No Starbucks logo or trademark identity assets included

Font instruction:
- Pretendard family only, per user instruction
- Deck text styles set with typeface "Pretendard"`;

const slides = [
  {
    number: "01",
    section: "FINAL PRESENTATION",
    title: "프랜차이즈 매장 테이블 청결도 자동 평가 프레임워크",
    subtitle: "CCTV와 메타데이터를 활용한 실전형 매장 수퍼바이징 접근",
    bullets: [
      "테이블 단위 청결도 1차 자동 평가",
      "비전 평가와 행동 평가를 결합한 구조",
      "최종 발표용 통합 데모 시스템 구현",
    ],
    notes: [
      "프로젝트 한 줄 소개부터 시작합니다.",
      "완전 자동 판정기보다는 본사 점검 부담을 줄이는 1차 자동 평가 프레임워크라는 점을 먼저 강조합니다.",
      "이후 슬라이드에서는 문제 정의, 구조, 구현, 결과, 한계 순으로 설명합니다.",
    ],
  },
  {
    number: "02",
    section: "BACKGROUND",
    title: "프로젝트 배경 및 문제 정의",
    bullets: [
      "위생 일관성은 브랜드 신뢰도와 고객 경험에 직접 연결",
      "가맹점 수가 늘수록 지점별 관리 편차 확대",
      "본사는 운영 전반을 자주 직접 확인하기 어려움",
      "결과적으로 점검 공백 시간이 누적",
    ],
    notes: [
      "이 슬라이드는 왜 이 문제가 중요한가를 설명하는 장표입니다.",
      "위생 문제는 단순 청소 이슈가 아니라 브랜드 경험 문제라는 점을 연결해서 말합니다.",
      "지점 수가 많을수록 사람 중심 점검만으로는 지속 관리가 어렵다는 흐름으로 이어갑니다.",
    ],
  },
  {
    number: "03",
    section: "LIMITS",
    title: "기존 방문 점검 방식의 한계",
    bullets: [
      "슈퍼바이저 방문 점검은 인력·시간 비용 큼",
      "방문 시점만 확인 가능",
      "비방문 시간대의 위생 상태 파악 어려움",
      "지속 모니터링 관점에서는 확장성 부족",
    ],
    notes: [
      "핵심은 기존 방식이 틀렸다는 것이 아니라, 자주 반복하기 어렵다는 점입니다.",
      "방문 간격 사이에 매장 상태가 어떻게 바뀌는지는 알기 어렵다는 운영 공백을 짚습니다.",
      "다음 슬라이드에서 그래서 CCTV와 메타데이터를 활용하자는 아이디어로 연결합니다.",
    ],
  },
  {
    number: "04",
    section: "IDEA",
    title: "핵심 아이디어",
    bullets: [
      "기존 CCTV와 테이블 위치 정보를 활용",
      "매장 전체가 아니라 테이블 단위로 평가",
      "보이는 흔적은 visual score로 반영",
      "청소 행위 정황은 action score로 보완",
      "최종적으로 운영 친화적인 청결 점수 산출",
    ],
    notes: [
      "이 프로젝트를 단일 이미지 분류가 아니라 근거 결합형 파이프라인으로 설명합니다.",
      "테이블 단위로 본다는 점이 중요한데, 실제 관리 단위를 자연스럽게 반영하기 때문입니다.",
      "visual과 action을 분리한 이유도 여기서 함께 설명합니다.",
    ],
  },
  {
    number: "05",
    section: "ARCHITECTURE",
    title: "전체 시스템 구조",
    bullets: [
      "CCTV 입력을 테이블 ROI 중심으로 해석",
      "객체·상태 분석 결과를 visual score로 변환",
      "식사 종료 이후 행동 정황을 action score로 변환",
      "두 근거를 결합해 final cleanliness score 생성",
      "결과는 리포트 형태로 저장·조회",
    ],
    notes: [
      "이 슬라이드는 발표 전체의 기준 구조입니다.",
      "뒤 슬라이드에서 설명하는 비전 평가, 액션 평가, 샘플링, 리포트가 이 구조의 어느 부분인지 계속 연결해 줍니다.",
      "코드 기준으로도 ROI 설정, workflow, hybrid scoring, reports가 모두 실제 연결되어 있습니다.",
    ],
  },
  {
    number: "06",
    section: "VISION",
    title: "테이블 단위 비전 청결도 평가",
    bullets: [
      "OpenCV 기반 ROI crop 후 테이블 영역만 입력",
      "OpenAI gpt-4.1-mini로 청결도 JSON 구조화 출력",
      "정확히 보이는 객체와 추정 객체를 분리 기록",
      "YOLOE (yoloe-26n-seg.pt) 결과는 보조 증거로 활용",
    ],
    notes: [
      "현재 비전 평가는 OpenCV로 ROI를 자른 뒤 OpenAI 모델에 구조화 JSON 응답을 요청하는 방식입니다.",
      "기본 모델 설정은 gpt-4.1-mini이며 exact_objects, estimated_objects, findings, score를 함께 받습니다.",
      "YOLO 보조 경로는 기본적으로 yoloe-26n-seg.pt를 사용하고, 코드상 fallback 이름은 yolov8s-worldv2.pt로 설정되어 있습니다.",
      "다만 Hugging Face revision hash나 YOLO checkpoint revision은 코드에 고정되어 있지 않으므로, 발표에서는 모델 ID와 weight 파일명을 기준으로 설명하는 편이 정직합니다.",
    ],
  },
  {
    number: "07",
    section: "ACTION",
    title: "Action 기반 청결도 보조 평가",
    bullets: [
      "Mask2Former (facebook/mask2former-swin-small-coco-instance) 사용",
      "점유 흐름으로 `CUSTOMER_IN_USE` → `MEAL_ENDED` 추정",
      "구역 체류 시간으로 `CLEANING_CANDIDATE` 판단",
      "전후 상태 변화가 보이면 `CLEANED_LIKELY`로 강화",
    ],
    notes: [
      "현재 person masking은 transformers 기반 Mask2Former 모델 `facebook/mask2former-swin-small-coco-instance`를 사용합니다.",
      "이 결과로 person_present와 person_count를 만들고, 시간 축에서 식사 종료와 청소 후보를 추정합니다.",
      "직원 체류만으로는 청소 완료를 단정하지 않고, 이후 시각 상태 개선이 함께 보여야 CLEANED_LIKELY로 가는 보수적 규칙 기반 구조입니다.",
    ],
  },
  {
    number: "08",
    section: "TIMING",
    title: "비디오 타이밍 추출 및 효율화",
    bullets: [
      "OpenCV로 비디오 프레임 샘플링 및 timestamp 부여",
      "이미지 시퀀스와 short video를 같은 workflow 입력으로 변환",
      "change score + person relevance로 의미 있는 시점 우선 선택",
      "결과적으로 workflow-from-images / workflow-from-video 모두 지원",
    ],
    notes: [
      "현재 전처리 모듈은 이미지 시퀀스와 비디오를 모두 workflow frame 리스트로 바꿉니다.",
      "dynamic sampling에서는 프레임 변화량과 테이블 주변 사람 관련성을 함께 보고, meal end나 cleaning candidate 같은 장면을 우선적으로 뽑습니다.",
      "이 장표에서는 단순 프레임 추출이 아니라 이벤트 중심 샘플링이라는 점을 짚어주면 좋습니다.",
    ],
  },
  {
    number: "09",
    section: "IMPLEMENTATION",
    title: "구현 구조 및 기술 스택",
    bullets: [
      "Backend: FastAPI + Uvicorn + python-multipart + Jinja2",
      "CV/ML: OpenCV, NumPy, Pillow, PyTorch, transformers, ultralytics, OpenAI",
      "Storage/UI: SQLite 기반 결과 저장, HTML/CSS/JS 리포트 화면",
      "Mobile wrapper: Capacitor Android 연동",
      "Pretrained model: gpt-4.1-mini, mask2former-swin-small-coco-instance, yoloe-26n-seg.pt",
    ],
    notes: [
      "이 슬라이드는 실제 사용 라이브러리와 모듈 구성을 한 번에 정리하는 장표입니다.",
      "Backend는 FastAPI 중심이고, 시각 처리 쪽은 OpenCV, NumPy, PyTorch, transformers, ultralytics, OpenAI SDK가 결합되어 있습니다.",
      "결과 저장은 SQLite를 사용하며, 최종 판단은 visual과 action을 50 대 50으로 결합한 뒤 운영상 보수적 cap rule을 적용하는 구조라고 설명하면 좋습니다.",
      "Pretrained 모델 버전 표기는 코드에 박혀 있는 모델명과 weight 파일명 기준으로 적었습니다.",
    ],
  },
  {
    number: "10",
    section: "DEMO",
    title: "구현 결과 및 시연 흐름",
    bullets: [
      "ROI 설정 후 이미지·비디오 기반 workflow 실행",
      "테이블별 visual score / action score / final score 산출",
      "이유 코드와 설명 문장까지 함께 확인 가능",
      "결과는 reports 화면에서 다시 조회 가능",
    ],
    notes: [
      "라이브 데모가 짧다면 Action Workflow Demo와 Reports를 중심으로 보여주면 됩니다.",
      "결과가 단순 점수 하나가 아니라 상태, reason codes, explanation까지 포함된다는 점이 해석 가능성을 높입니다.",
      "현재 구현된 기능과 확장 예정 기능을 구분해서 신뢰감 있게 설명하는 것이 중요합니다.",
    ],
  },
  {
    number: "11",
    section: "VALUE",
    title: "기대 효과",
    bullets: [
      "본사 입장: 점검 부담 완화",
      "운영 입장: 비방문 시간대 모니터링 보완",
      "매장 입장: 테이블 단위 관리 기준 정착 가능",
      "고객 입장: 더 일관된 식사 환경 기대",
    ],
    notes: [
      "여기서는 기술 그 자체보다 운영 가치에 초점을 둡니다.",
      "완전 자동 대체가 아니라 1차 자동 점검 도구라는 표현이 가장 설득력 있습니다.",
      "장기적으로는 위생뿐 아니라 매장 운영 전반을 보는 감독 시스템으로 확장 가능하다고 연결합니다.",
    ],
  },
  {
    number: "12",
    section: "LIMITS & NEXT",
    title: "한계점 및 개선 방향",
    bullets: [
      "CCTV 각도·가림·해상도에 따라 판정 안정성 차이",
      "얼룩·먼지·닦임 여부 직접 탐지는 아직 제한적",
      "직원·고객 구분과 이벤트 해석은 더 고도화 필요",
      "장기적으로 운영 데이터와 결합한 통합 수퍼바이징 확장 가능",
    ],
    notes: [
      "한계점을 숨기기보다 현재 범위를 명확히 규정하는 장표입니다.",
      "직접 탐지가 어려운 청결 요소를 action 기반 보조 평가로 메우고 있지만, 완전한 해결은 아니라는 점을 설명합니다.",
      "결론에서는 이 프로젝트가 실전형 출발점이며 확장 여지가 크다는 메시지로 마무리합니다.",
    ],
  },
];

function resolvePaths(overrides = {}) {
  const finalPptx = overrides.finalPptx ?? defaultFinalPptx;
  const workRoot = overrides.workRoot ?? defaultWorkRoot;
  const tmpDir = path.join(workRoot, "tmp");
  return {
    finalPptx,
    workRoot,
    tmpDir,
    previewDir: path.join(tmpDir, "preview"),
    layoutDir: path.join(tmpDir, "layout"),
    qaDir: path.join(tmpDir, "qa"),
    assetDir: path.join(tmpDir, "assets"),
  };
}

async function ensureDirs(paths) {
  const { finalPptx, previewDir, layoutDir, qaDir, assetDir } = paths;
  await fs.mkdir(path.dirname(finalPptx), { recursive: true });
  await fs.mkdir(previewDir, { recursive: true });
  await fs.mkdir(layoutDir, { recursive: true });
  await fs.mkdir(qaDir, { recursive: true });
  await fs.mkdir(assetDir, { recursive: true });
}

async function writeBlob(targetPath, blob) {
  const arrayBuffer = await blob.arrayBuffer();
  await fs.writeFile(targetPath, new Uint8Array(arrayBuffer));
}

async function readImageBuffer(imagePath) {
  const bytes = await fs.readFile(imagePath);
  return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
}

function contentTypeFor(imagePath) {
  const ext = path.extname(imagePath).toLowerCase();
  if (ext === ".png") return "image/png";
  if (ext === ".webp") return "image/webp";
  return "image/jpeg";
}

function addShape(slide, config, text, textStyle) {
  const shape = slide.shapes.add(config);
  if (text !== undefined) {
    shape.text = text;
  }
  if (textStyle) {
    shape.text.style = textStyle;
  }
  return shape;
}

function addCard(slide, position, options = {}) {
  return slide.shapes.add({
    geometry: "roundRect",
    position,
    fill: options.fill ?? COLORS.white,
    line: { style: "solid", fill: options.line ?? COLORS.line, width: 1 },
    borderRadius: options.borderRadius ?? "rounded-2xl",
    shadow: options.shadow ?? "shadow-sm",
  });
}

function addFooter(slide, page) {
  addShape(
    slide,
    {
      geometry: "textbox",
      position: { left: 72, top: 678, width: 260, height: 18 },
      fill: "none",
      line: { style: "solid", fill: "none", width: 0 },
    },
    "Final SW Project Presentation",
    {
      typeface: FONT_FAMILY,
      fontSize: 11,
      color: COLORS.textSoft,
    },
  );
  addShape(
    slide,
    {
      geometry: "textbox",
      position: { left: 1130, top: 676, width: 78, height: 18 },
      fill: "none",
      line: { style: "solid", fill: "none", width: 0 },
    },
    page,
    {
      typeface: FONT_FAMILY,
      fontSize: 12,
      bold: true,
      color: COLORS.brandGreen,
      alignment: "right",
    },
  );
}

function addSectionLabel(slide, section, tone = "light") {
  const fill = tone === "dark" ? "rgba(255,255,255,0.16)" : COLORS.mint;
  const line = tone === "dark" ? "rgba(255,255,255,0.18)" : COLORS.accentGreen;
  const textColor = tone === "dark" ? COLORS.white : COLORS.accentGreen;
  const pill = addCard(
    slide,
    { left: frame.left, top: frame.top, width: 172, height: 32 },
    {
      fill,
      line,
      borderRadius: "rounded-full",
      shadow: "shadow-none",
    },
  );
  pill.text = section;
  pill.text.style = {
    typeface: FONT_FAMILY,
    fontSize: 12,
    bold: true,
    color: textColor,
    alignment: "center",
    verticalAlign: "middle",
  };
}

function addTitle(slide, title, tone = "light") {
  return addShape(
    slide,
    {
      geometry: "textbox",
      position: { left: frame.left, top: frame.top + 52, width: 760, height: 54 },
      fill: "none",
      line: { style: "solid", fill: "none", width: 0 },
    },
    title,
    {
      typeface: FONT_FAMILY,
      fontSize: 34,
      bold: true,
      color: tone === "dark" ? COLORS.white : COLORS.textStrong,
      lineSpacing: 1.05,
    },
  );
}

function addBullets(slide, bullets, options = {}) {
  const left = options.left ?? frame.left;
  const top = options.top ?? frame.top + 136;
  const width = options.width ?? 480;
  const lineHeight = options.lineHeight ?? 74;
  const bulletColor = options.bulletColor ?? COLORS.accentGreen;
  const textColor = options.textColor ?? COLORS.textStrong;
  const fontSize = options.fontSize ?? 22;

  bullets.forEach((bullet, index) => {
    addShape(
      slide,
      {
        geometry: "ellipse",
        position: { left, top: top + index * lineHeight + 9, width: 12, height: 12 },
        fill: bulletColor,
        line: { style: "solid", fill: bulletColor, width: 0 },
      },
    );
    addShape(
      slide,
      {
        geometry: "textbox",
        position: { left: left + 24, top: top + index * lineHeight, width, height: 56 },
        fill: "none",
        line: { style: "solid", fill: "none", width: 0 },
      },
      bullet,
      {
        typeface: FONT_FAMILY,
        fontSize,
        color: textColor,
        lineSpacing: 1.18,
      },
    );
  });
}

function addNotes(slide, notes) {
  slide.speakerNotes.textFrame.setText(notes);
  slide.speakerNotes.setVisible(true);
}

function addCoverSlide(presentation, slideData) {
  const slide = presentation.slides.add();
  slide.background.fill = COLORS.deepGreen;

  addShape(
    slide,
    {
      geometry: "rect",
      position: { left: 0, top: 0, width: 1280, height: 720 },
      fill: "linear(180deg, #1E3932 0%, #25483f 100%)",
      line: { style: "solid", fill: "none", width: 0 },
    },
  );

  addShape(
    slide,
    {
      geometry: "roundRect",
      position: { left: 68, top: 74, width: 500, height: 546 },
      fill: "rgba(255,255,255,0.06)",
      line: { style: "solid", fill: "rgba(255,255,255,0.12)", width: 1 },
      borderRadius: "rounded-3xl",
      shadow: "shadow-none",
    },
  );

  addShape(
    slide,
    {
      geometry: "roundRect",
      position: { left: 100, top: 110, width: 156, height: 34 },
      fill: "rgba(255,255,255,0.12)",
      line: { style: "solid", fill: "rgba(255,255,255,0.16)", width: 1 },
      borderRadius: "rounded-full",
      shadow: "shadow-none",
    },
    slideData.section,
    {
      typeface: FONT_FAMILY,
      fontSize: 12,
      bold: true,
      color: COLORS.white,
      alignment: "center",
      verticalAlign: "middle",
    },
  );

  addShape(
    slide,
    {
      geometry: "textbox",
      position: { left: 100, top: 178, width: 620, height: 150 },
      fill: "none",
      line: { style: "solid", fill: "none", width: 0 },
    },
    slideData.title,
    {
      typeface: FONT_FAMILY,
      fontSize: 39,
      bold: true,
      color: COLORS.white,
      lineSpacing: 1.08,
    },
  );

  addShape(
    slide,
    {
      geometry: "textbox",
      position: { left: 102, top: 346, width: 510, height: 60 },
      fill: "none",
      line: { style: "solid", fill: "none", width: 0 },
    },
    slideData.subtitle,
    {
      typeface: FONT_FAMILY,
      fontSize: 19,
      color: "rgba(255,255,255,0.82)",
      lineSpacing: 1.25,
    },
  );

  slideData.bullets.forEach((bullet, index) => {
    addShape(
      slide,
      {
        geometry: "roundRect",
        position: { left: 102, top: 446 + index * 54, width: 450, height: 38 },
        fill: "rgba(255,255,255,0.1)",
        line: { style: "solid", fill: "rgba(255,255,255,0.08)", width: 1 },
        borderRadius: "rounded-full",
        shadow: "shadow-none",
      },
      bullet,
      {
        typeface: FONT_FAMILY,
        fontSize: 16,
        color: COLORS.white,
        alignment: "center",
        verticalAlign: "middle",
      },
    );
  });

  const panel = addCard(
    slide,
    { left: 770, top: 102, width: 400, height: 494 },
    {
      fill: COLORS.cream,
      line: "rgba(0,0,0,0.04)",
      borderRadius: "rounded-3xl",
      shadow: "shadow-md",
    },
  );

  addShape(
    slide,
    {
      geometry: "textbox",
      position: { left: 814, top: 144, width: 270, height: 36 },
      fill: "none",
      line: { style: "solid", fill: "none", width: 0 },
    },
    "평가 파이프라인",
    {
      typeface: FONT_FAMILY,
      fontSize: 22,
      bold: true,
      color: COLORS.brandGreen,
    },
  );

  const steps = [
    "CCTV 입력",
    "테이블 ROI 추출",
    "visual / action 분석",
    "최종 점수 산출",
    "리포트 저장",
  ];
  steps.forEach((step, index) => {
    const top = 204 + index * 64;
    addShape(
      slide,
      {
        geometry: "roundRect",
        position: { left: 818, top, width: 304, height: 42 },
        fill: index % 2 === 0 ? COLORS.softCard : COLORS.ceramic,
        line: { style: "solid", fill: COLORS.line, width: 1 },
        borderRadius: "rounded-full",
        shadow: "shadow-none",
      },
      step,
      {
        typeface: FONT_FAMILY,
        fontSize: 17,
        bold: index === steps.length - 1,
        color: COLORS.textStrong,
        alignment: "center",
        verticalAlign: "middle",
      },
    );
    if (index < steps.length - 1) {
      addShape(
        slide,
        {
          geometry: "downArrow",
          position: { left: 953, top: top + 45, width: 36, height: 20 },
          fill: COLORS.accentGreen,
          line: { style: "solid", fill: COLORS.accentGreen, width: 0 },
        },
      );
    }
  });

  addFooter(slide, slideData.number);
  addNotes(slide, slideData.notes);
}

function addStandardSlide(presentation, slideData, builder) {
  const slide = presentation.slides.add();
  slide.background.fill = COLORS.cream;
  addSectionLabel(slide, slideData.section);
  addTitle(slide, slideData.title);
  builder(slide, slideData);
  addFooter(slide, slideData.number);
  addNotes(slide, slideData.notes);
}

function addProblemSlide(slide, slideData) {
  addBullets(slide, slideData.bullets, { width: 540 });

  addCard(slide, { left: 744, top: 152, width: 390, height: 360 }, { fill: COLORS.white });
  addShape(
    slide,
    {
      geometry: "textbox",
      position: { left: 784, top: 188, width: 300, height: 34 },
      fill: "none",
      line: { style: "solid", fill: "none", width: 0 },
    },
    "운영 현장에서의 문제",
    {
      typeface: FONT_FAMILY,
      fontSize: 21,
      bold: true,
      color: COLORS.brandGreen,
    },
  );

  const rows = [
    ["본사", "모든 매장을 항상 직접 보기 어려움"],
    ["가맹점", "정리 편차가 누적될 수 있음"],
    ["고객", "지점마다 위생 경험이 달라질 수 있음"],
  ];
  rows.forEach(([label, text], index) => {
    const y = 250 + index * 86;
    addShape(
      slide,
      {
        geometry: "roundRect",
        position: { left: 788, top: y, width: 78, height: 32 },
        fill: COLORS.mint,
        line: { style: "solid", fill: COLORS.accentGreen, width: 0 },
        borderRadius: "rounded-full",
        shadow: "shadow-none",
      },
      label,
      {
        typeface: FONT_FAMILY,
        fontSize: 14,
        bold: true,
        color: COLORS.accentGreen,
        alignment: "center",
        verticalAlign: "middle",
      },
    );
    addShape(
      slide,
      {
        geometry: "textbox",
        position: { left: 788, top: y + 42, width: 286, height: 38 },
        fill: "none",
        line: { style: "solid", fill: "none", width: 0 },
      },
      text,
      {
        typeface: FONT_FAMILY,
        fontSize: 16,
        color: COLORS.textSoft,
        lineSpacing: 1.2,
      },
    );
  });
}

function addLimitsSlide(slide, slideData) {
  addBullets(slide, slideData.bullets, { width: 520 });

  addCard(slide, { left: 744, top: 146, width: 398, height: 376 }, { fill: COLORS.white });
  addShape(
    slide,
    {
      geometry: "textbox",
      position: { left: 786, top: 184, width: 280, height: 34 },
      fill: "none",
      line: { style: "solid", fill: "none", width: 0 },
    },
    "방문 점검 타임라인",
    {
      typeface: FONT_FAMILY,
      fontSize: 21,
      bold: true,
      color: COLORS.brandGreen,
    },
  );

  addShape(
    slide,
    {
      geometry: "line",
      position: { left: 806, top: 344, width: 260, height: 0 },
      fill: "none",
      line: { style: "solid", fill: COLORS.brandGreen, width: 2 },
    },
  );

  const checkpoints = [
    { left: 828, label: "방문" },
    { left: 932, label: "공백" },
    { left: 1034, label: "방문" },
  ];
  checkpoints.forEach((item, index) => {
    addShape(
      slide,
      {
        geometry: "ellipse",
        position: { left: item.left, top: 333, width: 22, height: 22 },
        fill: index === 1 ? COLORS.gold : COLORS.accentGreen,
        line: { style: "solid", fill: index === 1 ? COLORS.gold : COLORS.accentGreen, width: 0 },
      },
    );
    addShape(
      slide,
      {
        geometry: "textbox",
        position: { left: item.left - 18, top: 366, width: 58, height: 22 },
        fill: "none",
        line: { style: "solid", fill: "none", width: 0 },
      },
      item.label,
      {
        typeface: FONT_FAMILY,
        fontSize: 13,
        color: COLORS.textSoft,
        alignment: "center",
      },
    );
  });

  addShape(
    slide,
    {
      geometry: "textbox",
      position: { left: 790, top: 426, width: 288, height: 60 },
      fill: "none",
      line: { style: "solid", fill: "none", width: 0 },
    },
    "방문 사이 시간대는\n상태 추적이 어려움",
    {
      typeface: FONT_FAMILY,
      fontSize: 18,
      bold: true,
      color: COLORS.deepGreen,
      alignment: "center",
      lineSpacing: 1.15,
    },
  );
}

function addIdeaSlide(slide, slideData) {
  addBullets(slide, slideData.bullets, { width: 500 });

  const centerX = 886;
  const centerY = 330;
  const main = addCard(
    slide,
    { left: centerX - 120, top: centerY - 56, width: 240, height: 112 },
    { fill: COLORS.deepGreen, line: COLORS.deepGreen, shadow: "shadow-md" },
  );
  main.text = "테이블 단위\n청결도 평가";
  main.text.style = {
    typeface: FONT_FAMILY,
    fontSize: 24,
    bold: true,
    color: COLORS.white,
    alignment: "center",
    verticalAlign: "middle",
    lineSpacing: 1.12,
  };

  const nodes = [
    { x: 770, y: 190, text: "CCTV\n영상", fill: COLORS.white, color: COLORS.textStrong },
    { x: 1006, y: 190, text: "테이블 위치\n메타데이터", fill: COLORS.white, color: COLORS.textStrong },
    { x: 770, y: 448, text: "visual\n평가", fill: COLORS.mint, color: COLORS.accentGreen },
    { x: 1006, y: 448, text: "action\n평가", fill: COLORS.mint, color: COLORS.accentGreen },
  ];

  nodes.forEach((node) => {
    const card = addCard(
      slide,
      { left: node.x - 88, top: node.y - 44, width: 176, height: 88 },
      { fill: node.fill, line: COLORS.line, shadow: "shadow-none" },
    );
    card.text = node.text;
    card.text.style = {
      typeface: FONT_FAMILY,
      fontSize: 18,
      bold: true,
      color: node.color,
      alignment: "center",
      verticalAlign: "middle",
      lineSpacing: 1.15,
    };
  });
}

function addArchitectureSlide(slide, slideData) {
  addBullets(slide, slideData.bullets, { width: 450, lineHeight: 66, fontSize: 20 });

  const labels = [
    "CCTV 입력",
    "테이블 ROI 추출",
    "객체 / 상태 분석",
    "visual score",
    "action score",
    "final score",
    "report",
  ];
  const startLeft = 548;
  const top = 268;
  const width = 88;
  const gap = 10;
  labels.forEach((label, index) => {
    const left = startLeft + index * (width + gap);
    const box = addCard(
      slide,
      { left, top, width, height: 88 },
      {
        fill: index >= 3 && index <= 5 ? COLORS.mint : COLORS.white,
        line: COLORS.line,
        shadow: "shadow-none",
      },
    );
    box.text = label;
    box.text.style = {
      typeface: FONT_FAMILY,
      fontSize: label === "객체 / 상태 분석" ? 14 : 15,
      bold: true,
      color: index === 6 ? COLORS.accentGreen : COLORS.textStrong,
      alignment: "center",
      verticalAlign: "middle",
      lineSpacing: 1.15,
    };
    if (index < labels.length - 1) {
      addShape(
        slide,
        {
          geometry: "rightArrow",
          position: { left: left + width + 2, top: top + 28, width: 28, height: 28 },
          fill: COLORS.accentGreen,
          line: { style: "solid", fill: COLORS.accentGreen, width: 0 },
        },
      );
    }
  });
}

async function addVisionSlide(slide, slideData) {
  addBullets(slide, slideData.bullets, { width: 450 });

  const refBytes = await readImageBuffer(sourceImagePaths.reference);
  const cropBytes = await readImageBuffer(sourceImagePaths.crop);

  addCard(slide, { left: 660, top: 148, width: 500, height: 400 }, { fill: COLORS.white });
  slide.images.add({
    blob: refBytes,
    contentType: contentTypeFor(sourceImagePaths.reference),
    alt: "Reference CCTV frame",
    fit: "cover",
    geometry: "roundRect",
    borderRadius: "rounded-2xl",
    position: { left: 692, top: 198, width: 226, height: 284 },
  });
  slide.images.add({
    blob: cropBytes,
    contentType: contentTypeFor(sourceImagePaths.crop),
    alt: "Table crop sample",
    fit: "cover",
    geometry: "roundRect",
    borderRadius: "rounded-2xl",
    position: { left: 938, top: 198, width: 188, height: 284 },
  });

  addShape(
    slide,
    {
      geometry: "textbox",
      position: { left: 710, top: 496, width: 188, height: 20 },
      fill: "none",
      line: { style: "solid", fill: "none", width: 0 },
    },
    "원본 CCTV 프레임",
    {
      typeface: FONT_FAMILY,
      fontSize: 13,
      color: COLORS.textSoft,
      alignment: "center",
    },
  );
  addShape(
    slide,
    {
      geometry: "textbox",
      position: { left: 948, top: 496, width: 172, height: 20 },
      fill: "none",
      line: { style: "solid", fill: "none", width: 0 },
    },
    "테이블 crop 예시",
    {
      typeface: FONT_FAMILY,
      fontSize: 13,
      color: COLORS.textSoft,
      alignment: "center",
    },
  );
}

function addActionSlide(slide, slideData) {
  addBullets(slide, slideData.bullets, { width: 470 });

  const states = [
    "CUSTOMER\nIN USE",
    "MEAL\nENDED",
    "CLEANING\nCANDIDATE",
    "CLEANED\nLIKELY",
  ];
  states.forEach((state, index) => {
    const left = 742 + index * 102;
    const fill = index === states.length - 1 ? COLORS.deepGreen : index === 2 ? COLORS.mint : COLORS.white;
    const color = index === states.length - 1 ? COLORS.white : COLORS.textStrong;
    const card = addCard(
      slide,
      { left, top: 290, width: 126, height: 84 },
      { fill, line: COLORS.line, shadow: "shadow-none" },
    );
    card.text = state;
    card.text.style = {
      typeface: FONT_FAMILY,
      fontSize: 16,
      bold: true,
      color,
      alignment: "center",
      verticalAlign: "middle",
      lineSpacing: 1.15,
    };
    if (index < states.length - 1) {
      addShape(
        slide,
        {
          geometry: "rightArrow",
          position: { left: left + 127, top: 320, width: 22, height: 20 },
          fill: COLORS.accentGreen,
          line: { style: "solid", fill: COLORS.accentGreen, width: 0 },
        },
      );
    }
  });

  addShape(
    slide,
    {
      geometry: "textbox",
      position: { left: 760, top: 404, width: 386, height: 58 },
      fill: "none",
      line: { style: "solid", fill: "none", width: 0 },
    },
    "person_present / person_count / zone dwell / before-after state change",
    {
      typeface: FONT_FAMILY,
      fontSize: 15,
      color: COLORS.textSoft,
      alignment: "center",
      lineSpacing: 1.18,
    },
  );
}

async function addTimingSlide(slide, slideData) {
  addBullets(slide, slideData.bullets, { width: 450 });

  const frameA = await readImageBuffer(sourceImagePaths.frameA);
  const frameB = await readImageBuffer(sourceImagePaths.frameB);
  const frameC = await readImageBuffer(sourceImagePaths.frameC);

  addShape(
    slide,
    {
      geometry: "line",
      position: { left: 700, top: 420, width: 390, height: 0 },
      fill: "none",
      line: { style: "solid", fill: COLORS.brandGreen, width: 3 },
    },
  );

  const nodes = [
    { left: 734, top: 272, label: "초기 상태", bytes: frameA, path: sourceImagePaths.frameA },
    { left: 884, top: 210, label: "후보 시점", bytes: frameB, path: sourceImagePaths.frameB },
    { left: 1034, top: 272, label: "청소 후 확인", bytes: frameC, path: sourceImagePaths.frameC },
  ];
  nodes.forEach((node) => {
    slide.images.add({
      blob: node.bytes,
      contentType: contentTypeFor(node.path),
      alt: node.label,
      fit: "cover",
      geometry: "roundRect",
      borderRadius: "rounded-xl",
      position: { left: node.left, top: node.top, width: 108, height: 108 },
    });
    addShape(
      slide,
      {
        geometry: "ellipse",
        position: { left: node.left + 42, top: 409, width: 24, height: 24 },
        fill: COLORS.accentGreen,
        line: { style: "solid", fill: COLORS.accentGreen, width: 0 },
      },
    );
    addShape(
      slide,
      {
        geometry: "textbox",
        position: { left: node.left - 8, top: 446, width: 128, height: 22 },
        fill: "none",
        line: { style: "solid", fill: "none", width: 0 },
      },
      node.label,
      {
        typeface: FONT_FAMILY,
        fontSize: 13,
        color: COLORS.textSoft,
        alignment: "center",
      },
    );
  });
}

function addImplementationSlide(slide, slideData) {
  addBullets(slide, slideData.bullets, { width: 400, lineHeight: 54, fontSize: 18 });

  const cards = [
    { title: "Backend / API", items: ["FastAPI", "Uvicorn", "python-multipart", "Jinja2"] },
    { title: "CV / ML", items: ["gpt-4.1-mini", "mask2former-swin-small", "yoloe-26n-seg.pt", "OpenCV · PyTorch"] },
    { title: "Storage / UI", items: ["SQLite", "HTML / CSS / JS", "Reports dashboard", "Capacitor Android"] },
  ];
  cards.forEach((card, index) => {
    const left = 680 + index * 170;
    addCard(slide, { left, top: 188, width: 152, height: 310 }, { fill: COLORS.white });
    addShape(
      slide,
      {
        geometry: "textbox",
        position: { left: left + 14, top: 214, width: 124, height: 50 },
        fill: "none",
        line: { style: "solid", fill: "none", width: 0 },
      },
      card.title,
      {
        typeface: FONT_FAMILY,
        fontSize: 18,
        bold: true,
        color: COLORS.brandGreen,
        alignment: "center",
        lineSpacing: 1.15,
      },
    );
    card.items.forEach((item, itemIndex) => {
      addShape(
        slide,
        {
          geometry: "textbox",
          position: { left: left + 16, top: 284 + itemIndex * 46, width: 120, height: 34 },
          fill: "none",
          line: { style: "solid", fill: "none", width: 0 },
        },
        item,
        {
          typeface: FONT_FAMILY,
          fontSize: 14,
          color: COLORS.textStrong,
          alignment: "center",
          lineSpacing: 1.18,
        },
      );
    });
  });
}

function addDemoSlide(slide, slideData) {
  addBullets(slide, slideData.bullets, { width: 410, lineHeight: 58, fontSize: 20 });

  const flow = [
    "ROI 설정",
    "workflow 실행",
    "점수·상태 확인",
    "reports 저장",
  ];
  flow.forEach((item, index) => {
    const left = 694 + index * 110;
    const box = addCard(
      slide,
      { left, top: 240, width: 126, height: 84 },
      {
        fill: index === flow.length - 1 ? COLORS.deepGreen : COLORS.white,
        line: COLORS.line,
        shadow: "shadow-none",
      },
    );
    box.text = item;
    box.text.style = {
      typeface: FONT_FAMILY,
      fontSize: 17,
      bold: true,
      color: index === flow.length - 1 ? COLORS.white : COLORS.textStrong,
      alignment: "center",
      verticalAlign: "middle",
      lineSpacing: 1.15,
    };
    if (index < flow.length - 1) {
      addShape(
        slide,
        {
          geometry: "rightArrow",
          position: { left: left + 126, top: 270, width: 22, height: 20 },
          fill: COLORS.accentGreen,
          line: { style: "solid", fill: COLORS.accentGreen, width: 0 },
        },
      );
    }
  });

  addCard(slide, { left: 722, top: 390, width: 348, height: 114 }, { fill: COLORS.softCard, shadow: "shadow-none" });
  addShape(
    slide,
    {
      geometry: "textbox",
      position: { left: 748, top: 418, width: 290, height: 62 },
      fill: "none",
      line: { style: "solid", fill: "none", width: 0 },
    },
    "출력 예시\nmeal status / cleaning status / reason codes / explanation",
    {
      typeface: FONT_FAMILY,
      fontSize: 18,
      bold: true,
      color: COLORS.deepGreen,
      alignment: "center",
      lineSpacing: 1.2,
    },
  );
}

function addValueSlide(slide, slideData) {
  addBullets(slide, slideData.bullets, { width: 460 });

  const groups = [
    { title: "본사", body: "점검 부담 완화\n관리 공백 보완" },
    { title: "매장", body: "테이블 단위 관리 기준\n반복 점검 가능" },
    { title: "고객", body: "더 일관된\n식사 환경 기대" },
  ];
  groups.forEach((group, index) => {
    const left = 690 + index * 158;
    const box = addCard(
      slide,
      { left, top: 224, width: 142, height: 222 },
      { fill: index === 1 ? COLORS.mint : COLORS.white, line: COLORS.line, shadow: "shadow-none" },
    );
    box.text = `${group.title}\n\n${group.body}`;
    box.text.style = {
      typeface: FONT_FAMILY,
      fontSize: 17,
      bold: true,
      color: index === 1 ? COLORS.accentGreen : COLORS.textStrong,
      alignment: "center",
      verticalAlign: "middle",
      lineSpacing: 1.24,
    };
  });
}

function addClosingSlide(slide, slideData) {
  addBullets(slide, slideData.bullets, { width: 500 });

  addCard(slide, { left: 742, top: 174, width: 374, height: 330 }, { fill: COLORS.deepGreen, line: COLORS.deepGreen });
  addShape(
    slide,
    {
      geometry: "textbox",
      position: { left: 786, top: 228, width: 286, height: 50 },
      fill: "none",
      line: { style: "solid", fill: "none", width: 0 },
    },
    "핵심 결론",
    {
      typeface: FONT_FAMILY,
      fontSize: 24,
      bold: true,
      color: COLORS.white,
      alignment: "center",
    },
  );
  addShape(
    slide,
    {
      geometry: "textbox",
      position: { left: 792, top: 300, width: 274, height: 132 },
      fill: "none",
      line: { style: "solid", fill: "none", width: 0 },
    },
    "완전한 자동 판정기보다\n현실적인 1차 자동 점검 도구로 시작\n\n추후 통합 매장 수퍼바이징으로 확장 가능",
    {
      typeface: FONT_FAMILY,
      fontSize: 20,
      color: "rgba(255,255,255,0.9)",
      alignment: "center",
      lineSpacing: 1.28,
    },
  );
}

async function renderAndExport(presentation, paths) {
  const { previewDir, layoutDir, finalPptx } = paths;
  for (const [index, slide] of presentation.slides.items.entries()) {
    const stem = `slide-${String(index + 1).padStart(2, "0")}`;
    const png = await presentation.export({ slide, format: "png", scale: 1 });
    await writeBlob(path.join(previewDir, `${stem}.png`), png);
    const layout = await slide.export({ format: "layout" });
    await fs.writeFile(path.join(layoutDir, `${stem}.layout.json`), await layout.text(), "utf8");
  }
  const montage = await presentation.export({
    format: "png",
    montage: {
      format: "png",
      width: 1600,
      slideWidth: 380,
      padding: 24,
      gap: 20,
      background: "#f2f0eb",
      columns: 3,
    },
  });
  await writeBlob(path.join(previewDir, "deck-montage.png"), montage);

  const pptx = await PresentationFile.exportPptx(presentation);
  await pptx.save(finalPptx);
}

async function writeArtifacts(paths) {
  const { tmpDir, qaDir } = paths;
  await fs.writeFile(path.join(tmpDir, "slide-plan.txt"), slidePlanText, "utf8");
  await fs.writeFile(path.join(tmpDir, "source-notes.txt"), sourceNotesText, "utf8");
  await fs.writeFile(
    path.join(qaDir, "visual-qa.txt"),
    [
      "Visual QA checklist",
      "- Verify all 12 slides render",
      "- Check title alignment and footer consistency",
      "- Check Korean text overflow",
      "- Check Pretendard typeface appears in PPTX XML",
      "- Check embedded sample images appear on slides 06 and 08",
      "- Check final output path exists",
    ].join("\n"),
    "utf8",
  );
}

async function buildDeck(overrides = {}) {
  const paths = resolvePaths(overrides);
  await ensureDirs(paths);
  await writeArtifacts(paths);

  const presentation = Presentation.create({ slideSize });
  presentation.theme.colorScheme = {
    name: "Warm Green",
    themeColors: {
      accent1: COLORS.accentGreen,
      accent2: COLORS.brandGreen,
      accent3: COLORS.gold,
      accent4: COLORS.deepGreen,
      accent5: COLORS.mint,
      accent6: COLORS.ceramic,
      bg1: COLORS.white,
      bg2: COLORS.cream,
      tx1: COLORS.textStrong,
      tx2: COLORS.textSoft,
      dk1: "#000000",
      dk2: COLORS.deepGreen,
      lt1: COLORS.white,
      lt2: COLORS.ceramic,
      hlink: COLORS.accentGreen,
      folHlink: COLORS.brandGreen,
    },
  };

  addCoverSlide(presentation, slides[0]);
  addStandardSlide(presentation, slides[1], addProblemSlide);
  addStandardSlide(presentation, slides[2], addLimitsSlide);
  addStandardSlide(presentation, slides[3], addIdeaSlide);
  addStandardSlide(presentation, slides[4], addArchitectureSlide);
  addStandardSlide(presentation, slides[5], async () => {});
  addStandardSlide(presentation, slides[6], addActionSlide);
  addStandardSlide(presentation, slides[7], async () => {});
  addStandardSlide(presentation, slides[8], addImplementationSlide);
  addStandardSlide(presentation, slides[9], addDemoSlide);
  addStandardSlide(presentation, slides[10], addValueSlide);
  addStandardSlide(presentation, slides[11], addClosingSlide);

  await addVisionSlide(presentation.slides.getItem(5), slides[5]);
  await addTimingSlide(presentation.slides.getItem(7), slides[7]);

  await renderAndExport(presentation, paths);

  return {
    finalPptx: paths.finalPptx,
    workRoot: paths.workRoot,
    previewDir: paths.previewDir,
    layoutDir: paths.layoutDir,
    qaDir: paths.qaDir,
  };
}

export async function main(overrides = {}) {
  return buildDeck(overrides);
}

if (import.meta.url === `file://${__filename}`) {
  main()
    .then((result) => {
      console.log(JSON.stringify(result, null, 2));
    })
    .catch((error) => {
      console.error(error);
      process.exitCode = 1;
    });
}
