"""
main.py
FastAPI Application สำหรับวิเคราะห์หุ้นไทยสำหรับมือใหม่
Phase 1: Single Stock Analysis
Phase 2: Market Overview (^SET.BK) & Top Gainers
Phase 3: Seasonality & Sector Rotation
Phase 4: AI Stock Assistant (Gemini 2.5 Flash)
ใช้ yfinance, pandas, FastAPI, Uvicorn และ google-genai
"""

import sys
import os
import time
from typing import Dict, Any, Optional, List
import yfinance as yf
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from dotenv import load_dotenv

# โหลด Environment Variables จากไฟล์ .env
load_dotenv()

# =============================================================
# 🔑 [ใส่ API KEY ตรงนี้] บรรทัดที่ 28
# สามารถวาง Gemini API Key ของคุณลงในเครื่องหมายคำพูด "" ได้เลย
# หรือถ้ามีไฟล์ .env ก็จะโหลดมาใช้อัตโนมัติ
# =============================================================
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")  # <-- วาง Key ตรงนี้ เช่น "AIzaSy..."

# รองรับการแสดงผลภาษาไทยและ Emoji บน Windows Terminal
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# นำเข้า Google GenAI SDK อย่างปลอดภัย
try:
    from google import genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

# สร้าง FastAPI Instance
app = FastAPI(
    title="Thai Stock Analysis API",
    description="API สำหรับวิเคราะห์แนวโน้มหุ้นไทย, Seasonality, Sector Rotation และ AI Stock Assistant (Gemini)",
    version="4.0.0",
)

# เปิดใช้งาน CORS Middleware เพื่อรองรับการเชื่อมต่อจาก Frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory Caches เพื่อลด latency และหลีกเลี่ยงการยิง Yahoo Finance ซ้ำๆ
CACHE_TTL_SECONDS = 300

market_cache: Dict[str, Any] = {
    "timestamp": 0,
    "data": None,
}

sector_cache: Dict[str, Any] = {
    "timestamp": 0,
    "data": None,
}

seasonality_cache: Dict[str, Any] = {}


class AIConsultRequest(BaseModel):
    symbol: str
    question: str


def format_ticker_symbol(symbol: str) -> str:
    clean_symbol = symbol.strip().upper()
    if clean_symbol.startswith("^"):
        return clean_symbol
    return clean_symbol if clean_symbol.endswith(".BK") else f"{clean_symbol}.BK"


def analyze_thai_stock(symbol: str) -> Optional[Dict[str, Any]]:
    """
    วิเคราะห์หุ้นไทยย้อนหลัง 1 เดือน
    คำนวณผลตอบแทนรวม (%) และ Win Rate (% วันที่ราคาปิดบวก)
    """
    clean_symbol = symbol.strip().upper()
    ticker_symbol = format_ticker_symbol(symbol)

    ticker = yf.Ticker(ticker_symbol)
    df = ticker.history(period="1mo")

    if df.empty or len(df) < 2:
        return None

    initial_close = float(df["Close"].iloc[0])
    final_close = float(df["Close"].iloc[-1])
    total_return = ((final_close - initial_close) / initial_close) * 100

    daily_returns = df["Close"].pct_change().dropna()
    positive_days = int((daily_returns > 0).sum())
    total_days = int(len(daily_returns))
    win_rate = (positive_days / total_days) * 100 if total_days > 0 else 0.0

    if total_return > 2.0 and win_rate >= 55.0:
        status = "🟢 แนวโน้มแข็งแกร่ง (น่าสนใจ)"
    elif total_return < -2.0 or win_rate < 40.0:
        status = "🔴 แนวโน้มเป็นขาลง (ความเสี่ยงสูง)"
    else:
        status = "🟡 ผันผวน/ออกข้าง (ระมัดระวัง)"

    result = {
        "symbol": ticker_symbol,
        "raw_symbol": clean_symbol,
        "initial_close": round(initial_close, 2),
        "final_close": round(final_close, 2),
        "total_return": round(total_return, 2),
        "win_rate": round(win_rate, 2),
        "positive_days": positive_days,
        "total_days": total_days,
        "status": status,
    }

    return result


def get_market_overview() -> Dict[str, Any]:
    """
    ดึงข้อมูลดัชนีตลาดหุ้นไทย (^SET.BK) ย้อนหลัง 1 เดือน
    """
    ticker = yf.Ticker("^SET.BK")
    df = ticker.history(period="1mo")

    if len(df) < 5:
        df = ticker.history(period="3mo")
    if len(df) < 5:
        df = ticker.history(period="1y")
    if len(df) > 22:
        df = df.tail(22)

    if df.empty or len(df) < 2:
        return {
            "symbol": "^SET.BK",
            "name": "SET Index",
            "current_index": 0.0,
            "initial_index": 0.0,
            "total_return": 0.0,
            "win_rate": 0.0,
            "positive_days": 0,
            "total_days": 0,
            "signal": "yellow",
            "status": "🟡 ไม่สามารถดึงข้อมูลดัชนีได้",
            "summary": "ยังไม่สามารถประเมินภาพรวมตลาดได้ในขณะนี้",
        }

    initial_index = float(df["Close"].iloc[0])
    current_index = float(df["Close"].iloc[-1])
    total_return = ((current_index - initial_index) / initial_index) * 100

    daily_returns = df["Close"].pct_change().dropna()
    positive_days = int((daily_returns > 0).sum())
    total_days = int(len(daily_returns))
    win_rate = (positive_days / total_days) * 100 if total_days > 0 else 0.0

    if total_return > 2.0 and win_rate >= 55.0:
        signal = "green"
        status = "🟢 ตลาดแนวโน้มขาขึ้น (บรรยากาศสดใส)"
        summary = "ตลาดภาพรวมแข็งแกร่ง หุ้นส่วนใหญ่ปรับตัวขึ้น มีโอกาสลงทุนที่ดี"
    elif total_return < -2.0 or win_rate < 40.0:
        signal = "red"
        status = "🔴 ตลาดแนวโน้มขาลง (ความเสี่ยงสูง)"
        summary = "ตลาดอยู่ในช่วงปรับฐาน/ขาลง มีความเสี่ยงสูง ควรเน้นถือเงินสดหรือระมัดระวังเป็นพิเศษ"
    else:
        signal = "yellow"
        status = "🟡 ตลาดผันผวน/ออกข้าง (ระมัดระวัง)"
        summary = "ตลาดผันผวน ควรระมัดระวัง รอจังหวะเลือกหุ้นที่มีพื้นฐานหรือแนวโน้มเฉพาะตัว"

    return {
        "symbol": "^SET.BK",
        "name": "SET Index",
        "current_index": round(current_index, 2),
        "initial_index": round(initial_index, 2),
        "total_return": round(total_return, 2),
        "win_rate": round(win_rate, 2),
        "positive_days": positive_days,
        "total_days": total_days,
        "signal": signal,
        "status": status,
        "summary": summary,
    }


def calculate_seasonality(symbol: str) -> Optional[Dict[str, Any]]:
    """
    คำนวณสถิติย้อนหลัง 3 เดือน แยกตามวันในสัปดาห์ (จันทร์ - ศุกร์)
    """
    clean_symbol = symbol.strip().upper()
    ticker_symbol = format_ticker_symbol(symbol)

    ticker = yf.Ticker(ticker_symbol)
    df = ticker.history(period="3mo")

    if df.empty or len(df) < 5:
        df = ticker.history(period="6mo")
        if len(df) > 66:
            df = df.tail(66)

    if df.empty or len(df) < 5:
        return None

    df["Daily_Return"] = df["Close"].pct_change() * 100
    df["DayOfWeek"] = df.index.dayofweek

    days_metadata = [
        {"idx": 0, "name": "วันจันทร์", "short": "จันทร์"},
        {"idx": 1, "name": "วันอังคาร", "short": "อังคาร"},
        {"idx": 2, "name": "วันพุธ", "short": "พุธ"},
        {"idx": 3, "name": "วันพฤหัสบดี", "short": "พฤหัสฯ"},
        {"idx": 4, "name": "วันศุกร์", "short": "ศุกร์"},
    ]

    daily_stats: List[Dict[str, Any]] = []

    for day in days_metadata:
        day_df = df[df["DayOfWeek"] == day["idx"]].dropna(subset=["Daily_Return"])
        cnt = len(day_df)
        pos = int((day_df["Daily_Return"] > 0).sum())
        win_rate = (pos / cnt * 100) if cnt > 0 else 0.0
        avg_ret = float(day_df["Daily_Return"].mean()) if cnt > 0 else 0.0

        daily_stats.append({
            "day_idx": day["idx"],
            "day_name": day["name"],
            "short_name": day["short"],
            "total_days": cnt,
            "positive_days": pos,
            "win_rate": round(win_rate, 1),
            "avg_return": round(avg_ret, 2),
        })

    sorted_days = sorted(daily_stats, key=lambda x: (x["win_rate"], x["avg_return"]), reverse=True)
    top_days = [d for d in sorted_days if d["win_rate"] >= 50.0]

    if len(top_days) >= 2:
        best_day_names = [top_days[0]["day_name"], top_days[1]["day_name"]]
        summary = (
            f"{best_day_names[0]} และ {best_day_names[1]} "
            f"มักเป็นวันที่ราคาขยับขึ้นดีที่สุด (Win Rate {top_days[0]['win_rate']}% และ {top_days[1]['win_rate']}%)"
        )
    elif len(top_days) == 1:
        best_day_names = [top_days[0]["day_name"]]
        summary = (
            f"{best_day_names[0]} "
            f"มักเป็นวันที่ราคาขยับขึ้นดีที่สุด (Win Rate {top_days[0]['win_rate']}%, ผลตอบแทนเฉลี่ย {top_days[0]['avg_return']:+.2f}%)"
        )
    else:
        best = sorted_days[0]
        best_day_names = [best["day_name"]]
        summary = (
            f"{best['day_name']} เป็นวันที่มีผลการดำเนินงานดีที่สุดเมื่อเทียบกับวันอื่นในรอบ 3 เดือน (Win Rate {best['win_rate']}%)"
        )

    return {
        "symbol": ticker_symbol,
        "raw_symbol": clean_symbol,
        "period": "3mo",
        "best_days": best_day_names,
        "summary": summary,
        "daily_stats": daily_stats,
    }


@app.get("/robots.txt", include_in_schema=False)
def robots_txt():
    """
    คืนค่าไฟล์ robots.txt เพื่อป้องกัน Search Engine Crawlers (Google, Bing) แอบสแกนเว็บ
    """
    robots_path = os.path.join(os.path.dirname(__file__), "robots.txt")
    if not os.path.exists(robots_path):
        robots_path = "robots.txt"
    return FileResponse(robots_path, media_type="text/plain")


@app.get("/")
def root():
    """
    Health Check Endpoint สำหรับตรวจสอบสถานะของ API และคืนค่าข้อความต้อนรับ
    """
    return {
        "status": "online",
        "message": "ยินดีต้อนรับสู่ระบบวิเคราะห์หุ้นไทยสำหรับมือใหม่ (Thai Stock Analysis API - Phase 4 AI Assistant)",
        "docs_url": "/docs",
        "endpoints": {
            "health_check": "GET /",
            "market_summary": "GET /market-summary",
            "sector_rotation": "GET /sector-rotation",
            "seasonality": "GET /seasonality/{symbol}",
            "analyze_stock": "GET /analyze/{symbol}",
            "ai_consult": "POST /ai-consult",
        },
    }


@app.get("/market-summary")
def market_summary():
    """
    Endpoint สรุปภาพรวมตลาดหุ้นไทย (^SET.BK) และ 3 หุ้นดาวรุ่ง (Top Gainers)
    """
    current_time = time.time()
    if market_cache["data"] and (current_time - market_cache["timestamp"] < CACHE_TTL_SECONDS):
        return market_cache["data"]

    market_data = get_market_overview()

    popular_symbols = ["PTT", "CPALL", "DELTA", "AOT", "KBANK"]
    stock_results: List[Dict[str, Any]] = []

    for sym in popular_symbols:
        res = analyze_thai_stock(sym)
        if res:
            stock_results.append(res)

    stock_results.sort(key=lambda x: x["total_return"], reverse=True)
    top_3_gainers = stock_results[:3]

    for idx, item in enumerate(top_3_gainers, start=1):
        item["rank"] = idx

    response_data = {
        "market": market_data,
        "top_gainers": top_3_gainers,
        "all_popular": stock_results,
    }

    market_cache["data"] = response_data
    market_cache["timestamp"] = current_time

    return response_data


@app.get("/sector-rotation")
def sector_rotation():
    """
    Endpoint สำหรับดึงข้อมูลวงจรหมุนเวียนกลุ่มอุตสาหกรรม (Sector Rotation)
    """
    current_time = time.time()
    if sector_cache["data"] and (current_time - sector_cache["timestamp"] < CACHE_TTL_SECONDS):
        return sector_cache["data"]

    sector_definitions = [
        {
            "sector_id": "ENERG",
            "name": "Energy (พลังงาน)",
            "icon": "⚡",
            "stocks": ["PTT", "PTTEP"],
        },
        {
            "sector_id": "BANK",
            "name": "Banking (ธนาคาร)",
            "icon": "🏦",
            "stocks": ["KBANK", "SCB"],
        },
        {
            "sector_id": "COMM",
            "name": "Commerce (พาณิชย์)",
            "icon": "🛒",
            "stocks": ["CPALL", "CRC"],
        },
        {
            "sector_id": "ICT",
            "name": "ICT (สื่อสาร)",
            "icon": "📡",
            "stocks": ["ADVANC", "TRUE"],
        },
    ]

    sectors_result: List[Dict[str, Any]] = []

    for sec in sector_definitions:
        stock_details = []
        returns = []
        win_rates = []

        for sym in sec["stocks"]:
            analysis = analyze_thai_stock(sym)
            if analysis:
                stock_details.append({
                    "symbol": analysis["raw_symbol"],
                    "price": analysis["final_close"],
                    "return": analysis["total_return"],
                    "win_rate": analysis["win_rate"],
                    "status": analysis["status"],
                })
                returns.append(analysis["total_return"])
                win_rates.append(analysis["win_rate"])

        avg_return = round(sum(returns) / len(returns), 2) if returns else 0.0
        avg_win_rate = round(sum(win_rates) / len(win_rates), 1) if win_rates else 0.0

        if avg_return > 2.0:
            status = "🟢 เงินไหลเข้า (แข็งแกร่ง)"
            status_type = "bullish"
        elif avg_return < -2.0:
            status = "🔴 เงินไหลออก (ชะลอตัว)"
            status_type = "bearish"
        else:
            status = "🟡 ทรงตัว/พักฐาน"
            status_type = "neutral"

        sectors_result.append({
            "sector_id": sec["sector_id"],
            "name": sec["name"],
            "icon": sec["icon"],
            "representative_stocks": sec["stocks"],
            "stock_details": stock_details,
            "avg_return": avg_return,
            "avg_win_rate": avg_win_rate,
            "status": status,
            "status_type": status_type,
        })

    sectors_result.sort(key=lambda x: x["avg_return"], reverse=True)

    for idx, s in enumerate(sectors_result, start=1):
        s["rank"] = idx

    strongest = sectors_result[0]
    return_sign = "+" if strongest["avg_return"] >= 0 else ""
    summary_text = (
        f"กลุ่มอุตสาหกรรมที่แข็งแกร่งที่สุดในรอบ 1 เดือนคือ {strongest['name']} "
        f"(ผลตอบแทนเฉลี่ย {return_sign}{strongest['avg_return']}%) สัญญาณบ่งชี้ว่าเงินลงทุนกำลังไหลเข้ากลุ่มนี้เด่นชัด"
    )

    response_data = {
        "summary": summary_text,
        "strongest_sector": strongest["name"],
        "sectors": sectors_result,
    }

    sector_cache["data"] = response_data
    sector_cache["timestamp"] = current_time

    return response_data


@app.get("/seasonality/{symbol}")
def get_seasonality(symbol: str):
    """
    Endpoint คำนวณสถิติย้อนหลัง 3 เดือนว่าหุ้นตัวนี้หรือ SET มักจะให้ผลตอบแทนบวกดีที่สุดในวันใดของสัปดาห์
    """
    clean_sym = symbol.strip().upper()
    current_time = time.time()

    if clean_sym in seasonality_cache:
        cached_entry = seasonality_cache[clean_sym]
        if current_time - cached_entry["timestamp"] < CACHE_TTL_SECONDS:
            return cached_entry["data"]

    seasonality_data = calculate_seasonality(symbol)
    if not seasonality_data:
        raise HTTPException(
            status_code=404,
            detail=f"ไม่พบข้อมูลย้อนหลังเพียงพอสำหรับคำนวณ Seasonality ของหุ้น: '{symbol}'",
        )

    seasonality_cache[clean_sym] = {
        "timestamp": current_time,
        "data": seasonality_data,
    }

    return seasonality_data


@app.get("/analyze/{symbol}")
def analyze_stock(symbol: str):
    """
    Endpoint วิเคราะห์หุ้นไทยย้อนหลัง 1 เดือน (พ่วง Seasonality)
    """
    analysis = analyze_thai_stock(symbol)
    if not analysis:
        raise HTTPException(
            status_code=404,
            detail=f"ไม่พบข้อมูลราคาสำหรับหุ้น: '{symbol}' กรุณาตรวจสอบชื่อหุ้นอีกครั้ง",
        )

    seasonality = calculate_seasonality(symbol)
    if seasonality:
        analysis["seasonality"] = seasonality

    return analysis


@app.post("/ai-consult")
def ai_consult(payload: AIConsultRequest):
    """
    Endpoint AI Stock Assistant:
    รับค่าชื่อหุ้นและคำถามจากผู้ใช้
    ดึงสถิติหุ้นนั้นจาก yfinance แล้วส่ง prompt ให้โมเดล gemini-2.5-flash ช่วยวิเคราะห์คำตอบ
    """
    raw_symbol = payload.symbol.strip().upper()
    user_question = payload.question.strip()

    if not raw_symbol:
        raise HTTPException(status_code=400, detail="กรุณาระบุชื่อหุ้น")
    if not user_question:
        raise HTTPException(status_code=400, detail="กรุณาระบุคำถาม")

    # 1. ดึงสถิติหุ้นจาก yfinance
    stock_data = analyze_thai_stock(raw_symbol)
    if not stock_data:
        raise HTTPException(
            status_code=404,
            detail=f"ไม่พบข้อมูลราคาสำหรับหุ้น '{raw_symbol}' บน Yahoo Finance",
        )

    seasonality_data = calculate_seasonality(raw_symbol)
    seasonality_text = seasonality_data["summary"] if seasonality_data else "ไม่มีข้อมูล"

    # 2. เตรียม Context และสร้าง Prompt สำหรับ Gemini
    system_instructions = (
        "คุณคือ 'AI ผู้ช่วยวิเคราะห์หุ้นไทยสำหรับมือใหม่' (Thai Stock Investment Assistant) "
        "ตอบคำถามอย่างเป็นมิตร อธิบายให้มือใหม่เข้าใจง่าย ชัดเจน ตรงประเด็น ใช้ภาษาไทยที่สุภาพ "
        "อ้างอิงสถิติและตัวเลขจริงที่ได้รับใน Context และแนะนำหลักการบริหารความเสี่ยงเสมอ "
        "ห้ามการันตีผลกำไรหรือสั่งซื้อขายโดยตรง แต่ให้วิเคราะห์เชิงสถิติและข้อสังเกต "
        "จัดรูปแบบข้อความเป็น bullet points ให้อ่านง่ายและน่าติดตาม"
    )

    prompt = f"""
{system_instructions}

[ข้อมูลสถิติหุ้นจริงย้อนหลัง]:
- หุ้น: {stock_data['symbol']} (ชื่อย่อ: {stock_data['raw_symbol']})
- ราคาเริ่มต้นรอบ 1 เดือน: {stock_data['initial_close']} บาท
- ราคาปิดล่าสุด: {stock_data['final_close']} บาท
- ผลตอบแทนรอบ 1 เดือน: {stock_data['total_return']:+.2f}%
- อัตราปิดบวก (Win Rate): {stock_data['win_rate']}% (ปิดบวก {stock_data['positive_days']} จากทั้งหมด {stock_data['total_days']} วันทำการ)
- สถานะสัญญาณเบื้องต้น: {stock_data['status']}
- สถิติวันในสัปดาห์ (Seasonality 3 เดือน): {seasonality_text}

[คำถามจากผู้ใช้ (มือใหม่)]:
"{user_question}"

กรุณาตอบคำถามข้างต้น โดยวิเคราะห์เชื่อมโยงกับข้อมูลสถิติข้างต้น ให้ข้อแนะนำที่เป็นรูปธรรมสำหรับมือใหม่ และปิดท้ายด้วยคำเตือนเรื่องความเสี่ยงสั้นๆ
"""

    # 3. ตรวจสอบ API Key และเรียกใช้โมเดล gemini-2.5-flash
    api_key = GEMINI_API_KEY or os.environ.get("GEMINI_API_KEY")

    if not GENAI_AVAILABLE:
        raise HTTPException(
            status_code=500,
            detail="แพ็กเกจ google-genai ยังไม่ได้ติดตั้งในระบบ กรุณาติดตั้งผ่าน pip install google-genai",
        )

    if not api_key:
        # Fallback กรณีผู้ใช้ยังไม่ได้กำหนด GEMINI_API_KEY เพื่อให้ระบบยังตอบสนองได้
        fallback_answer = (
            f"⚠️ **ระบบตรวจไม่พบ GEMINI_API_KEY ใน Environment**\n\n"
            f"*(คำแนะนำ: กรุณากำหนด `GEMINI_API_KEY` ในไฟล์ `.env` หรือ Environment Variables เพื่อเปิดใช้งานการวิเคราะห์แบบเต็มประสิทธิภาพด้วยโมเดล `gemini-2.5-flash`)*\n\n"
            f"---\n\n"
            f"💡 **บทวิเคราะห์เบื้องต้นตามสถิติตัวเลขสำหรับหุ้น {stock_data['raw_symbol']}**:\n"
            f"- **ทิศทางราคาล่าสุด**: ปัจจุบันราคาอยู่ที่ **{stock_data['final_close']} บาท** (ผลตอบแทน 1 เดือน: **{stock_data['total_return']:+.2f}%**)\n"
            f"- **ความต่อเนื่องของการขึ้น (Win Rate)**: อยู่ที่ **{stock_data['win_rate']}%** ({stock_data['positive_days']}/{stock_data['total_days']} วันทำการ)\n"
            f"- **สถานะสำหรับมือใหม่**: {stock_data['status']}\n"
            f"- **วันทำการที่มักขยับขึ้นดี (Seasonality)**: {seasonality_text}\n\n"
            f"📌 **คำแนะนำเบื้องต้นสำหรับคำถาม '{user_question}'**:\n"
            f"1. หากสถานะเป็น 🟢 แข็งแกร่ง สามารถพิจารณารอจังหวะย่อตัวในวันทำการที่มีสถิติเด่นเพื่อเข้าสะสม\n"
            f"2. หากสถานะเป็น 🔴 ขาลง หรือ 🟡 ผันผวน แนะนำให้ตั้งจุดตัดขาดทุน (Stop loss) และแบ่งสัดส่วนเงินลงทุนให้เหมาะสม ไม่ควรรีบเข้าซื้อถัวเฉลี่ย\n\n"
            f"⚠️ *คำเตือน: การลงทุนมีความเสี่ยง ข้อมูลนี้เป็นการวิเคราะห์เชิงสถิติเบื้องต้นเท่านั้น ผู้ลงทุนควรศึกษาข้อมูลก่อนตัดสินใจ*"
        )
        return {
            "symbol": stock_data["raw_symbol"],
            "question": user_question,
            "answer": fallback_answer,
            "model": "fallback-heuristic (GEMINI_API_KEY not configured)",
        }

    try:
        client = genai.Client(api_key=api_key)
        model_name = "gemini-2.5-flash"
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
            )
        except Exception as model_err:
            # รองรับกรณี Google อัปเกรดโมเดลใหม่ หรือคีย์ใหม่ที่รองรับ gemini-3.6-flash
            if "404" in str(model_err) or "NOT_FOUND" in str(model_err) or "no longer available" in str(model_err):
                model_name = "gemini-3.6-flash"
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                )
            else:
                raise model_err

        answer_text = response.text or "ขออภัย ไม่สามารถสร้างคำตอบได้ในขณะนี้"

        return {
            "symbol": stock_data["raw_symbol"],
            "question": user_question,
            "answer": answer_text,
            "model": model_name,
        }

    except Exception as e:
        error_msg = str(e)
        raise HTTPException(
            status_code=500,
            detail=f"เกิดข้อผิดพลาดในการเชื่อมต่อกับ Gemini API: {error_msg}",
        )


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
