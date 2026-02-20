"""
Booking Engine for Bella Roma AI Restaurant Bot.
Handles table availability checks, reservations, and alternative suggestions.
"""

import json
import re
from pathlib import Path
from datetime import datetime


# Path to availability data
DATA_DIR = Path(__file__).parent / "data"
AVAILABILITY_PATH = DATA_DIR / "availability.json"

# Default date used when user does not specify one
DEFAULT_DATE = "2026-02-20"


class BookingEngine:
    """Rule-based booking engine for restaurant table reservations."""

    def __init__(self):
        """Load availability data from JSON file."""
        with open(AVAILABILITY_PATH, "r", encoding="utf-8") as f:
            self.availability = json.load(f)

    def check_availability(self, date: str, time: str) -> dict:
        """
        Check if tables are available for a given date and time.

        Args:
            date: Date string in YYYY-MM-DD format.
            time: Time string in HH:MM format.

        Returns:
            Dict with 'available' (bool) and 'tables' (int) keys.
        """
        if date not in self.availability:
            return {"available": False, "tables": 0, "reason": "date_not_found"}

        day_schedule = self.availability[date]
        if time not in day_schedule:
            return {"available": False, "tables": 0, "reason": "time_not_found"}

        tables = day_schedule[time]
        return {
            "available": tables > 0,
            "tables": tables,
            "reason": "ok" if tables > 0 else "fully_booked",
        }

    def book_table(self, date: str, time: str, guests: int) -> str:
        """
        Attempt to book a table for the given date, time, and guest count.

        Args:
            date: Date string in YYYY-MM-DD format.
            time: Time string in HH:MM format.
            guests: Number of guests (must be > 0).

        Returns:
            A user-friendly confirmation or error message.
        """
        # Validate guest count
        if guests <= 0:
            return "🚫 Number of guests must be at least 1. Please try again."

        # Check availability
        status = self.check_availability(date, time)

        if status["reason"] == "date_not_found":
            alternatives = self.suggest_alternative(date)
            return (
                f"🚫 Sorry, we don't have availability data for {self._format_date(date)}. "
                f"{alternatives}"
            )

        if status["reason"] == "time_not_found":
            alternatives = self.suggest_alternative(date)
            return (
                f"🚫 Sorry, we don't offer reservations at {time} on {self._format_date(date)}. "
                f"{alternatives}"
            )

        if not status["available"]:
            alternatives = self.suggest_alternative(date)
            return (
                f"🚫 Sorry, no tables are available at {time} on {self._format_date(date)}. "
                f"{alternatives}"
            )

        # Book the table — decrement availability
        self.availability[date][time] -= 1

        remaining = self.availability[date][time]
        return (
            f"✅ Reservation confirmed!\n\n"
            f"📅 Date: {self._format_date(date)}\n"
            f"🕐 Time: {time}\n"
            f"👥 Guests: {guests}\n\n"
            f"🍕 We look forward to welcoming you at Bella Roma! "
            f"({remaining} table{'s' if remaining != 1 else ''} remaining for this slot)"
        )

    def suggest_alternative(self, date: str) -> str:
        """
        Suggest alternative time slots for the given date, or alternative dates.

        Args:
            date: Date string in YYYY-MM-DD format.

        Returns:
            A suggestion message string.
        """
        # Try to find available slots on the same date
        if date in self.availability:
            day_schedule = self.availability[date]
            available_slots = [
                f"{t} ({c} table{'s' if c != 1 else ''} left)"
                for t, c in day_schedule.items()
                if c > 0
            ]
            if available_slots:
                slots_text = ", ".join(available_slots)
                return f"📋 Available times on {self._format_date(date)}: {slots_text}"

        # No slots on that date — suggest other dates
        other_dates = []
        for d, schedule in self.availability.items():
            if d != date:
                total_tables = sum(schedule.values())
                if total_tables > 0:
                    other_dates.append(self._format_date(d))

        if other_dates:
            dates_text = ", ".join(other_dates)
            return f"📋 No availability on {self._format_date(date)}. Try these dates instead: {dates_text}"

        return "😔 Unfortunately, we have no available tables at this time. Please try again later."

    def handle_message(self, message: str) -> str:
        """
        Parse a booking-related message and route to the appropriate method.

        Extracts date, time, and guest count from natural language input.

        Args:
            message: The user's booking-related message.

        Returns:
            A response string.
        """
        message_lower = message.lower()

        # Extract date (YYYY-MM-DD pattern)
        date_match = re.search(r"(\d{4}-\d{2}-\d{2})", message)
        date = date_match.group(1) if date_match else DEFAULT_DATE

        # Extract time (HH:MM pattern)
        time_match = re.search(r"(\d{1,2}:\d{2})", message)
        time_str = None
        if time_match:
            raw_time = time_match.group(1)
            # Normalize to HH:MM
            parts = raw_time.split(":")
            time_str = f"{int(parts[0]):02d}:{parts[1]}"

        # Extract guest count
        guest_match = re.search(r"(\d+)\s*(?:guest|people|person|pax|seat)", message_lower)
        guests = int(guest_match.group(1)) if guest_match else 0

        # Also try patterns like "for 4" or "table for 2"
        if guests == 0:
            for_match = re.search(r"(?:for|of)\s+(\d+)", message_lower)
            if for_match:
                guests = int(for_match.group(1))

        # Determine intent
        if any(word in message_lower for word in ["check", "available", "availability", "open"]):
            if time_str:
                status = self.check_availability(date, time_str)
                if status["available"]:
                    return (
                        f"✅ Yes! We have {status['tables']} table{'s' if status['tables'] != 1 else ''} "
                        f"available at {time_str} on {self._format_date(date)}. "
                        f"Would you like to book one?"
                    )
                else:
                    alternatives = self.suggest_alternative(date)
                    return f"🚫 No tables available at {time_str} on {self._format_date(date)}. {alternatives}"
            else:
                return self.suggest_alternative(date)

        if any(word in message_lower for word in ["book", "reserve", "reservation"]):
            if not time_str:
                return (
                    "🍕 I'd love to help you book a table! "
                    "Please provide the time (e.g., 19:00), "
                    "number of guests (e.g., 4 guests), "
                    "and optionally a date (e.g., 2026-02-20). "
                    "Default date is February 20, 2026."
                )
            if guests == 0:
                return (
                    f"🍕 Great choice! {time_str} on {self._format_date(date)} — "
                    f"how many guests will be joining? "
                    f"(e.g., 'book for 4 guests at {time_str}')"
                )
            return self.book_table(date, time_str, guests)

        # Fallback — general booking help
        return (
            "🍕 I can help you with reservations! Try:\n\n"
            "• \"Book a table for 4 at 19:00\"\n"
            "• \"Check availability at 20:00\"\n"
            "• \"Reserve for 2 guests at 18:00 on 2026-02-21\"\n\n"
            f"Our available dates: {', '.join(self._format_date(d) for d in self.availability.keys())}"
        )

    def _format_date(self, date_str: str) -> str:
        """Format a YYYY-MM-DD date string into a readable format."""
        try:
            dt = datetime.strptime(date_str, "%Y-%m-%d")
            return dt.strftime("%B %d, %Y")
        except ValueError:
            return date_str
