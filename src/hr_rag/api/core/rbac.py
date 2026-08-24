# who can see what. compensation has salary bands / bonus formulas, so it's
# scoped to manager+ instead of every employee
ROLE_ALLOWED_CATEGORIES = {
    "employee": ["leave", "conduct", "recruitment", "it", "operations"],
    "manager": ["leave", "conduct", "recruitment", "performance", "it", "operations"],
    "finance_user": ["leave", "conduct", "recruitment", "finance", "it", "operations"],
    "hr_admin": [
        "leave", "conduct", "recruitment", "performance", "compensation",
        "finance", "it", "legal", "operations",
    ],
}

VALID_ROLES = list(ROLE_ALLOWED_CATEGORIES.keys())


def allowed_categories_for_role(role: str) -> list:
    return ROLE_ALLOWED_CATEGORIES.get(role, [])
