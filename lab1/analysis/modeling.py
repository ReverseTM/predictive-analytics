"""Подбор моделей и оценка прогноза для месячных продаж."""

import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX


def fit_model(y, x, order, seasonal_order, likelihood_start=None):
    """ARIMA - частный случай SARIMAX без сезонности и внешних признаков."""
    # Годовая разность обнуляет повторяющийся декабрьский индикатор.
    if x is not None and seasonal_order[1] == 1:
        x = x.drop(columns="HolidayMonth")

    # Тысячи вместо единиц помогают оптимизатору; прогноз вернём в исходный масштаб.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = SARIMAX(
            y / 1000,
            exog=x,
            order=order,
            seasonal_order=seasonal_order,
            trend="c",
            enforce_stationarity=True,
            enforce_invertibility=True,
            simple_differencing=False,
        )
        if likelihood_start is not None:
            # Первые месяцы обновляют фильтр, но не входят в целевую сумму.
            model.loglikelihood_burn = likelihood_start
        result = model.fit(disp=False, maxiter=500)

    messages = "; ".join(sorted({str(w.message) for w in caught}))

    return result, messages


def metrics(actual, predicted):
    """MSE в квадрате исходных единиц; R² относительно среднего проверяемого отрезка."""
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    mse = np.mean((actual - predicted) ** 2)
    total = np.sum((actual - actual.mean()) ** 2)
    r2 = 1 - np.sum((actual - predicted) ** 2) / total if total > 0 else np.nan

    return {"MSE": mse, "R²": r2}


def select_models(y, x):
    """Подбираем небольшую сетку только на первых 36 месяцах."""
    rows, predictions = [], {}

    # В каждом срезе прогноз строится сразу на весь оставшийся отрезок до теста.

    origins = (24, 27, 30)
    orders = [
        (0, 0, 0),
        (1, 0, 0),
        (2, 0, 0),
        (0, 0, 1),
        (1, 0, 1),
        (0, 1, 0),
        (1, 1, 0),
        (0, 1, 1),
    ]
    seasonal_orders = [(1, 0, 0, 12), (0, 0, 1, 12), (0, 1, 0, 12)]

    for family in ("ARIMA", "SARIMAX"):
        seasons = [(0, 0, 0, 0)] if family == "ARIMA" else seasonal_orders
        for order in orders:
            for season in seasons:
                key = (family, order, season)
                fold_rows, messages, converged = [], [], True

                try:
                    for origin in origins:
                        exog = x.iloc[:origin] if family == "SARIMAX" else None
                        future_x = x.iloc[origin:36] if family == "SARIMAX" else None
                        fitted, message = fit_model(
                            y.iloc[:origin], exog, order, season
                        )
                        messages.append(message)
                        converged &= bool(fitted.mle_retvals.get("converged", False))

                        if future_x is not None:
                            future_x = future_x[fitted.model.exog_names]

                        forecast = fitted.get_forecast(36 - origin, exog=future_x)
                        pred = np.asarray(forecast.predicted_mean) * 1000

                        for h, value in enumerate(pred, 1):
                            fold_rows.append(
                                {
                                    "origin": origin,
                                    "h": h,
                                    "actual": y.iloc[origin + h - 1],
                                    "pred": value,
                                    "naive": y.iloc[origin + h - 13],
                                    "scale": y.iloc[:origin].mean(),
                                }
                            )

                    folds = pd.DataFrame(fold_rows)
                    score = metrics(folds.actual, folds.pred)
                    predictions[key] = folds
                    rows.append(
                        {
                            "Модель": family,
                            "order": order,
                            "seasonal_order": season,
                            **score,
                            "Сходимость": converged,
                            "Предупреждения": " | ".join(
                                sorted(set(filter(None, messages)))
                            ),
                        }
                    )
                except (ValueError, np.linalg.LinAlgError) as error:
                    rows.append(
                        {
                            "Модель": family,
                            "order": order,
                            "seasonal_order": season,
                            "MSE": np.nan,
                            "R²": np.nan,
                            "Сходимость": False,
                            "Предупреждения": str(error),
                        }
                    )

    table = pd.DataFrame(rows).sort_values("MSE")
    best = {}

    for family in ("ARIMA", "SARIMAX"):
        eligible = table[
            (table["Модель"] == family) & table["Сходимость"] & table.MSE.notna()
        ]
        if eligible.empty:
            raise RuntimeError(f"Нет сошедшихся кандидатов для {family}")

        row = eligible.iloc[0]
        best[family] = (family, row.order, row.seasonal_order)

    return table, best, predictions


def horizon_quality(folds):
    """Качество всего прогноза от первого месяца до выбранного горизонта."""
    rows = []
    for h in range(3, 13):
        part = folds[folds.h <= h]
        score = metrics(part.actual, part.pred)
        baseline = metrics(part.actual, part.naive)["MSE"]
        relative_rmse = np.sqrt(np.mean(((part.actual - part.pred) / part.scale) ** 2))

        rows.append(
            {
                "Горизонт": h,
                **score,
                "MSE сезонного повтора": baseline,
                "Относительная RMSE": relative_rmse,
                "Приемлемо": score["R²"] >= 0
                and score["MSE"] <= baseline
                and relative_rmse <= 0.20,
                "Число прогнозов": len(part),
                "Прогнозов на шаге h": int((folds.h == h).sum()),
            }
        )

    return pd.DataFrame(rows).set_index("Горизонт")


def supported_horizon(table):
    # Оцениваем весь отрезок до h, независимо от оценок коротких отрезков.
    accepted = table.index[table["Приемлемо"]]
    return int(accepted.max()) if len(accepted) else 0
