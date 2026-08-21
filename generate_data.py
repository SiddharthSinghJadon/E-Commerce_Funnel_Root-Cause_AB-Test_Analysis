"""
Generate simulated e-commerce session and A/B test datasets.

The project uses synthetic data because a public dataset does not provide
both session-level funnel events and randomized A/B-test assignments together.

The generator intentionally seeds two known effects:
1. Mobile sessions have lower cart-to-checkout conversion.
2. The one-page checkout treatment has a positive checkout-to-order lift.

These known effects allow the analysis pipeline to be checked against
expected signals before using the workflow on real data.
"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_SEED = 42

N_SESSIONS = 52_000
START_DATE = datetime(2026, 4, 1)
DAYS = 60

DEVICES = ["mobile", "desktop", "tablet"]
DEVICE_WEIGHTS = [0.62, 0.30, 0.08]

SOURCES = [
    "organic",
    "paid_social",
    "paid_search",
    "direct",
    "email",
]
SOURCE_WEIGHTS = [0.34, 0.22, 0.20, 0.16, 0.08]

BASE_RATES = {
    "open_to_search": 0.72,
    "search_to_view": 0.66,
    "view_to_cart": 0.34,
    "cart_to_checkout": 0.52,
    "checkout_to_order": 0.79,
}

DEVICE_CART_CHECKOUT_ADJ = {
    "mobile": -0.13,
    "desktop": +0.05,
    "tablet": -0.02,
}

N_AB = 9_000
AB_BASE_RATE = 0.79
AB_TREATMENT_LIFT = 0.055


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def hour_adjustment(hour):
    """Return the late-night checkout conversion adjustment."""
    return -0.04 if (hour >= 23 or hour <= 5) else 0.0


def generate_sessions(rng):
    """Generate the session-level e-commerce funnel dataset."""
    rows = []

    for session_id in range(1, N_SESSIONS + 1):
        device = rng.choice(DEVICES, p=DEVICE_WEIGHTS)
        source = rng.choice(SOURCES, p=SOURCE_WEIGHTS)

        day_offset = rng.integers(0, DAYS)
        hour = int(rng.normal(15, 5)) % 24

        timestamp = (
            START_DATE
            + timedelta(
                days=int(day_offset),
                hours=hour,
                minutes=int(rng.integers(0, 60)),
            )
        )

        opened = True

        searched = (
            rng.random() < BASE_RATES["open_to_search"]
        )

        viewed = (
            searched
            and rng.random() < BASE_RATES["search_to_view"]
        )

        carted = (
            viewed
            and rng.random() < BASE_RATES["view_to_cart"]
        )

        cart_checkout_probability = (
            BASE_RATES["cart_to_checkout"]
            + DEVICE_CART_CHECKOUT_ADJ[device]
            + hour_adjustment(hour)
        )

        cart_checkout_probability = min(
            max(cart_checkout_probability, 0.02),
            0.98,
        )

        checked_out = (
            carted
            and rng.random() < cart_checkout_probability
        )

        ordered = (
            checked_out
            and rng.random() < BASE_RATES["checkout_to_order"]
        )

        rows.append(
            [
                session_id,
                timestamp,
                device,
                source,
                opened,
                searched,
                viewed,
                carted,
                checked_out,
                ordered,
            ]
        )

    columns = [
        "session_id",
        "timestamp",
        "device",
        "traffic_source",
        "opened",
        "searched",
        "viewed_product",
        "added_to_cart",
        "reached_checkout",
        "placed_order",
    ]

    return pd.DataFrame(rows, columns=columns)


def generate_ab_test(rng):
    """Generate randomized control and treatment observations."""
    arm = rng.choice(
        ["control", "treatment"],
        size=N_AB,
        p=[0.5, 0.5],
    )

    converted = np.array(
        [
            rng.random()
            < (
                AB_BASE_RATE
                + (
                    AB_TREATMENT_LIFT
                    if assignment == "treatment"
                    else 0.0
                )
            )
            for assignment in arm
        ]
    )

    device = rng.choice(
        DEVICES,
        size=N_AB,
        p=DEVICE_WEIGHTS,
    )

    return pd.DataFrame(
        {
            "user_id": np.arange(1, N_AB + 1),
            "arm": arm,
            "device": device,
            "converted": converted.astype(int),
        }
    )


# ============================================================
# MAIN
# ============================================================

def main():
    """Generate and save both project datasets."""
    rng = np.random.default_rng(RANDOM_SEED)

    sessions = generate_sessions(rng)
    sessions.to_csv("data/sessions.csv", index=False)

    print("sessions.csv:", sessions.shape)
    print(
        sessions[
            [
                "opened",
                "searched",
                "viewed_product",
                "added_to_cart",
                "reached_checkout",
                "placed_order",
            ]
        ].sum()
    )

    ab = generate_ab_test(rng)
    ab.to_csv("data/ab_test_checkout.csv", index=False)

    print("\nab_test_checkout.csv:", ab.shape)
    print(
        ab.groupby("arm")["converted"]
        .agg(["mean", "count"])
    )


if __name__ == "__main__":
    main()
