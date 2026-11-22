import pytest

from src.config import CURRENCY_SYMBOL, quote_table_name, validate_table_name


def test_currency_is_inr():
    assert CURRENCY_SYMBOL == "₹"


@pytest.mark.parametrize("name", ["sales", "Sales_Data", "public.sales", "_s1"])
def test_valid_table_names(name):
    assert validate_table_name(name) == name


@pytest.mark.parametrize(
    "name",
    ["", "  ", "sales; DROP TABLE sales", "sales--", "sales table", "123abc", 'sales"', "sales."],
)
def test_invalid_table_names_rejected(name):
    with pytest.raises(ValueError):
        validate_table_name(name)


def test_quote_table_name_quotes_each_part():
    assert quote_table_name("public.sales") == '"public"."sales"'
    assert quote_table_name("sales") == '"sales"'
