# NOTE: the tests do not import pytest at module level, see conftest.py
try:
    from ..cron import parse_cron_expression
except ImportError:  # pytest (see conftest.py)
    from autochunk_plugin.cron import parse_cron_expression


def _raises_value_error(expression) -> bool:
    try:
        parse_cron_expression(expression)
    except ValueError:
        return True
    return False


def test_parse_basic_expression():
    assert parse_cron_expression("0 3 * * 0") == {
        "minute": "0", "hour": "3", "day": "*", "month": "*", "day_of_week": "sun",
    }


def test_parse_every_quarter_of_hour():
    assert parse_cron_expression("*/15 * * * *")["minute"] == "*/15"
    assert parse_cron_expression("*/15 * * * *")["day_of_week"] == "*"


def test_crontab_day_of_week_numbering():
    # crontab: 0 and 7 are Sunday, 1 is Monday (APScheduler would read 0 as Monday)
    assert parse_cron_expression("0 0 * * 7")["day_of_week"] == "sun"
    assert parse_cron_expression("0 0 * * 1")["day_of_week"] == "mon"
    assert parse_cron_expression("0 0 * * 1-5")["day_of_week"] == "mon,tue,wed,thu,fri"
    assert parse_cron_expression("0 0 * * 5-7")["day_of_week"] == "fri,sat,sun"
    assert parse_cron_expression("0 0 * * */2")["day_of_week"] == "sun,tue,thu,sat"
    assert parse_cron_expression("0 0 * * 1,3")["day_of_week"] == "mon,wed"


def test_named_days_are_kept():
    assert parse_cron_expression("0 0 * * mon-fri")["day_of_week"] == "mon-fri"
    assert parse_cron_expression("0 0 * * SAT")["day_of_week"] == "sat"


def test_invalid_expressions():
    assert _raises_value_error("* * *")
    assert _raises_value_error("* * * * * *")
    assert _raises_value_error("61 * * * *")
    assert _raises_value_error("0 25 * * *")
    assert _raises_value_error("0 0 * * 8")
    assert _raises_value_error("0 0 * * 5-2")
    assert _raises_value_error(None)
