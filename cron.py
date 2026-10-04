from typing import Dict

_DOW_NAMES = ["sun", "mon", "tue", "wed", "thu", "fri", "sat"]


def _convert_dow_token(token: str) -> str:
    """Convert a crontab day-of-week token (0/7 = Sunday) into APScheduler-compatible names.

    APScheduler 3 numbers the days of week from Monday (0 = mon), unlike crontab (0 = sun): numeric days, ranges and
    steps are expanded into explicit day names, which mean the same in both. Named tokens (e.g. "mon-fri") are kept.
    """
    token = token.strip().lower()
    step = 1
    if "/" in token:
        token, raw_step = token.split("/", 1)
        if not raw_step.isdigit() or int(raw_step) < 1:
            raise ValueError(f"Invalid day-of-week step: {raw_step}")
        step = int(raw_step)

    if token in ("*", "?"):
        if step == 1:
            return "*"
        start, end = 0, 6
    elif "-" in token:
        start_s, end_s = token.split("-", 1)
        if not (start_s.isdigit() and end_s.isdigit()):
            # named range: same semantics in crontab and APScheduler
            return token if step == 1 else f"{token}/{step}"
        start, end = int(start_s), int(end_s)
    elif token.isdigit():
        start = end = int(token)
    else:
        if step != 1:
            raise ValueError(f"Invalid day-of-week token: {token}/{step}")
        return token

    if not (0 <= start <= 7 and 0 <= end <= 7 and start <= end):
        raise ValueError(f"Invalid day of week range: {start}-{end}")

    names = []
    for number in range(start, end + 1, step):
        name = _DOW_NAMES[number % 7]
        if name not in names:
            names.append(name)
    return ",".join(names)


def parse_cron_expression(expression: str) -> Dict[str, str]:
    """Parse a standard 5-field cron expression into the keyword arguments of ``WhiteRabbit.schedule_cron_job``.

    Args:
        expression: "minute hour day-of-month month day-of-week", e.g. "0 3 * * 0".

    Returns:
        The kwargs ``minute``, ``hour``, ``day``, ``month``, ``day_of_week``.

    Raises:
        ValueError: if the expression is not valid.
    """
    if not isinstance(expression, str):
        raise ValueError("The cron expression must be a string")

    fields = expression.split()
    if len(fields) != 5:
        raise ValueError(
            f"Invalid cron expression '{expression}': expected 5 fields (minute hour day month day-of-week)"
        )

    minute, hour, day, month, day_of_week = fields
    day_of_week = ",".join(_convert_dow_token(token) for token in day_of_week.split(","))

    result = {"minute": minute, "hour": hour, "day": day, "month": month, "day_of_week": day_of_week}

    # validate through APScheduler itself, the scheduler used by White Rabbit
    try:
        from apscheduler.triggers.cron import CronTrigger
    except ImportError:  # White Rabbit (and APScheduler with it) not installed: syntax-only check
        return result

    try:
        CronTrigger(**result, timezone="UTC")
    except Exception as e:
        raise ValueError(f"Invalid cron expression '{expression}': {e}") from e

    return result
