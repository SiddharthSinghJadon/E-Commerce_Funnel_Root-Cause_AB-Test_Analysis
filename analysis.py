import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = "data"
CHARTS_DIR = "charts"

BLUE = "#1f4e79"
RED = "#c0392b"
GREEN = "#1f8a4c"
GREY = "#7f7f7f"


# ============================================================
# DATA LOADING
# ============================================================

def load_data():
    """Load the session and A/B test datasets."""
    sessions = pd.read_csv(
        f"{DATA_DIR}/sessions.csv",
        parse_dates=["timestamp"]
    )
    ab = pd.read_csv(f"{DATA_DIR}/ab_test_checkout.csv")
    return sessions, ab


# ============================================================
# 1. FUNNEL ANALYSIS
# ============================================================

def analyze_funnel(sessions):
    """Calculate funnel stage counts and step-wise conversion rates."""
    stages = [
        "opened",
        "searched",
        "viewed_product",
        "added_to_cart",
        "reached_checkout",
        "placed_order",
    ]

    stage_labels = [
        "App Open",
        "Search",
        "Product View",
        "Add to Cart",
        "Checkout",
        "Order Placed",
    ]

    counts = [sessions[stage].sum() for stage in stages]
    step_conv = [
        100.0
    ] + [
        round(counts[i] / counts[i - 1] * 100, 1)
        for i in range(1, len(counts))
    ]

    overall_conv = round(counts[-1] / counts[0] * 100, 2)

    print("=== FUNNEL ===")
    for label, count, conversion in zip(
        stage_labels, counts, step_conv
    ):
        print(f"{label:15s} {count:6d}   step-conv: {conversion}%")

    print(f"Overall open->order conversion: {overall_conv}%")

    drop = [
        (
            stage_labels[i - 1],
            stage_labels[i],
            round(100 - step_conv[i], 1),
        )
        for i in range(1, len(step_conv))
    ]
    biggest = max(drop, key=lambda x: x[2])
    print("Biggest single-stage drop-off:", biggest)

    return stage_labels, counts, step_conv


def plot_funnel(stage_labels, counts, step_conv):
    """Create and save the funnel chart."""
    fig, ax = plt.subplots(figsize=(9, 5.5))

    bars = ax.bar(stage_labels, counts, color=BLUE)
    ax.set_title(
        "E-Commerce Funnel: Sessions by Stage (n=52,000 sessions)",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_ylabel("Sessions")
    ax.set_ylim(0, max(counts) * 1.28)

    for i, (bar, count) in enumerate(zip(bars, counts)):
        ax.annotate(
            f"{count:,}",
            (bar.get_x() + bar.get_width() / 2, count),
            textcoords="offset points",
            xytext=(0, 6),
            ha="center",
            fontsize=10,
            fontweight="bold",
        )

        if i > 0:
            ax.annotate(
                f"step-conv {step_conv[i]}%",
                (bar.get_x() + bar.get_width() / 2, count),
                textcoords="offset points",
                xytext=(0, 22),
                ha="center",
                fontsize=9,
                color=RED,
            )

    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(f"{CHARTS_DIR}/1_funnel.png", dpi=150)
    plt.close()


# ============================================================
# 2. ROOT CAUSE ANALYSIS: CART -> CHECKOUT
# ============================================================

def analyze_rca(sessions):
    """Analyze cart-to-checkout conversion by device and time of day."""
    carted = sessions[sessions["added_to_cart"]].copy()

    by_device = (
        carted.groupby("device")["reached_checkout"]
        .mean()
        .mul(100)
        .sort_values()
    )

    print("\n=== RCA: cart->checkout conversion by device ===")
    print(by_device.round(1))

    carted["hour"] = carted["timestamp"].dt.hour
    carted["is_late_night"] = carted["hour"].apply(
        lambda hour:
            "Late night (11pm-5am)"
            if (hour >= 23 or hour <= 5)
            else "Rest of day"
    )

    by_time = (
        carted.groupby("is_late_night")["reached_checkout"]
        .mean()
        .mul(100)
    )

    print("\n=== RCA: cart->checkout conversion by time of day ===")
    print(by_time.round(1))

    return carted, by_device, by_time


def plot_rca(by_device, by_time):
    """Create and save RCA segmentation charts."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    colors_d = [
        RED if device == "mobile"
        else GREEN if device == "desktop"
        else GREY
        for device in by_device.index
    ]

    axes[0].bar(by_device.index, by_device.values, color=colors_d)
    axes[0].set_title(
        "Cart→Checkout Conversion\nby Device",
        fontsize=11,
        fontweight="bold",
    )
    axes[0].set_ylabel("Conversion %")

    for i, value in enumerate(by_device.values):
        axes[0].annotate(
            f"{value:.1f}%",
            (i, value),
            textcoords="offset points",
            xytext=(0, 4),
            ha="center",
        )

    colors_t = [RED, GREEN]

    axes[1].bar(by_time.index, by_time.values, color=colors_t)
    axes[1].set_title(
        "Cart→Checkout Conversion\nby Time of Day",
        fontsize=11,
        fontweight="bold",
    )
    axes[1].set_ylabel("Conversion %")

    plt.setp(
        axes[1].get_xticklabels(),
        rotation=10,
        ha="right",
    )

    for i, value in enumerate(by_time.values):
        axes[1].annotate(
            f"{value:.1f}%",
            (i, value),
            textcoords="offset points",
            xytext=(0, 4),
            ha="center",
        )

    plt.tight_layout()
    plt.savefig(f"{CHARTS_DIR}/2_rca_segments.png", dpi=150)
    plt.close()


def test_mobile_vs_desktop(carted):
    """Run a two-proportion z-test for mobile vs desktop conversion."""
    mobile = carted[carted["device"] == "mobile"]
    desktop = carted[carted["device"] == "desktop"]

    p1 = mobile["reached_checkout"].mean()
    n1 = len(mobile)

    p2 = desktop["reached_checkout"].mean()
    n2 = len(desktop)

    pooled_p = (
        mobile["reached_checkout"].sum()
        + desktop["reached_checkout"].sum()
    ) / (n1 + n2)

    se = (
        pooled_p * (1 - pooled_p)
        * (1 / n1 + 1 / n2)
    ) ** 0.5

    z = (p1 - p2) / se
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))

    print(
        f"\nMobile vs Desktop cart->checkout: "
        f"{p1 * 100:.1f}% vs {p2 * 100:.1f}% "
        f"(n={n1},{n2}), z={z:.2f}, p={p_value:.2e}"
    )

    return z, p_value


# ============================================================
# 3. A/B TEST: SIMPLIFIED CHECKOUT
# ============================================================

def analyze_ab_test(ab):
    """Calculate A/B test conversion, lift, z-statistic and p-value."""
    group = (
        ab.groupby("arm")["converted"]
        .agg(["mean", "sum", "count"])
    )

    p_control = group.loc["control", "mean"]
    n_control = group.loc["control", "count"]

    p_treatment = group.loc["treatment", "mean"]
    n_treatment = group.loc["treatment", "count"]

    pooled_p = (
        group.loc["control", "sum"]
        + group.loc["treatment", "sum"]
    ) / (n_control + n_treatment)

    se = (
        pooled_p * (1 - pooled_p)
        * (1 / n_control + 1 / n_treatment)
    ) ** 0.5

    z = (p_treatment - p_control) / se
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))

    relative_lift = (
        (p_treatment - p_control) / p_control * 100
    )

    print("\n=== A/B TEST: Simplified checkout ===")
    print(f"Control:   {p_control * 100:.2f}% (n={n_control})")
    print(
        f"Treatment: {p_treatment * 100:.2f}% "
        f"(n={n_treatment})"
    )
    print(
        f"Absolute lift: {(p_treatment - p_control) * 100:.2f}pp "
        f"| Relative lift: {relative_lift:.1f}%"
    )
    print(f"z={z:.2f}, p={p_value:.4f}")

    return {
        "control_conversion": p_control,
        "treatment_conversion": p_treatment,
        "n_control": n_control,
        "n_treatment": n_treatment,
        "absolute_lift": p_treatment - p_control,
        "relative_lift": relative_lift,
        "z": z,
        "p_value": p_value,
    }


def plot_ab_test(results):
    """Create and save the A/B test chart."""
    control = results["control_conversion"] * 100
    treatment = results["treatment_conversion"] * 100
    p_value = results["p_value"]
    total_n = results["n_control"] + results["n_treatment"]

    fig, ax = plt.subplots(figsize=(6, 4.5))

    bars = ax.bar(
        ["Control\n(multi-step)", "Treatment\n(1-page checkout)"],
        [control, treatment],
        color=[GREY, BLUE],
    )

    ax.set_title(
        f"A/B Test: Checkout→Order Conversion\n"
        f"(p={p_value:.4f}, n={total_n:,})",
        fontsize=11,
        fontweight="bold",
    )
    ax.set_ylabel("Conversion %")
    ax.set_ylim(0, 100)

    for bar, value in zip(bars, [control, treatment]):
        ax.annotate(
            f"{value:.1f}%",
            (bar.get_x() + bar.get_width() / 2, value),
            textcoords="offset points",
            xytext=(0, 5),
            ha="center",
            fontsize=11,
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig(f"{CHARTS_DIR}/3_ab_test.png", dpi=150)
    plt.close()


# ============================================================
# 4. WEEKLY CONVERSION TREND
# ============================================================

def analyze_weekly_trend(sessions):
    """Calculate weekly open-to-order conversion."""
    sessions = sessions.copy()
    sessions["week"] = (
        sessions["timestamp"]
        .dt.to_period("W")
        .apply(lambda period: period.start_time)
    )

    weekly = (
        sessions.groupby("week")
        .apply(
            lambda group:
                group["placed_order"].sum()
                / group["opened"].sum()
                * 100
        )
    )

    return weekly


def plot_weekly_trend(weekly):
    """Create and save the weekly conversion trend chart."""
    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(
        weekly.index,
        weekly.values,
        marker="o",
        color=BLUE,
    )
    ax.set_title(
        "Weekly Open→Order Conversion Rate",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_ylabel("Conversion %")
    ax.grid(alpha=0.3)

    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(f"{CHARTS_DIR}/4_weekly_trend.png", dpi=150)
    plt.close()


# ============================================================
# MAIN
# ============================================================

def main():
    """Run the complete e-commerce funnel analysis."""
    sessions, ab = load_data()

    stage_labels, counts, step_conv = analyze_funnel(sessions)
    plot_funnel(stage_labels, counts, step_conv)

    carted, by_device, by_time = analyze_rca(sessions)
    plot_rca(by_device, by_time)
    test_mobile_vs_desktop(carted)

    ab_results = analyze_ab_test(ab)
    plot_ab_test(ab_results)

    weekly = analyze_weekly_trend(sessions)
    plot_weekly_trend(weekly)

    print("\nAll charts saved to charts/")


if __name__ == "__main__":
    main()
