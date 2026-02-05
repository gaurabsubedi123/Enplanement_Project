"""
LAS Airport Enplanement Forecasting - 20 Year Outlook (2025-2045)
Author: Gaurab Subedi
Date: January 2025

This script compares 8 different forecasting models (ML + econometric) 
for Harry Reid International Airport. Trying to figure out which approach 
works best with limited historical data.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import RobustScaler, PolynomialFeatures
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("LAS AIRPORT FORECASTING: 2025-2045")
print("="*70)

# Load the data
print("\n[1] Loading historical data...")
excel_file = 'Yearly_Enplanments_data__Harry_reid_airport_.xlsx'
df = pd.read_excel(excel_file)

# Just keep 2000-2024 data
df = df[df['years'].between(2000, 2024)].copy()
df['years'] = df['years'].astype(int)
print(f"    Got {len(df)} years of data (2000-2024)")

# COVID years messed everything up, so we're excluding them
# Otherwise models completely break (tried it, doesn't work)
covid_years = [2020, 2021, 2022]
df_train = df[~df['years'].isin(covid_years)].copy()
df_train = df_train.dropna(subset=['enplanments', 'clark-county_population', 'real_gdp_per_person'])

print(f"    Training on {len(df_train)} years (dropped COVID: {covid_years})")
print(f"    Years used: {sorted(df_train['years'].unique())}")

# Future population and GDP projections
# These come from BEA/CBO/Nevada State Demographer forecasts
print("\n[2] Setting up future projections...")

# Population projections through 2045
pop_forecast = {
    2025: 2443000, 2026: 2493000, 2027: 2537000, 2028: 2578000, 2029: 2617000,
    2030: 2655000, 2031: 2692000, 2032: 2728000, 2033: 2764000, 2034: 2797000,
    2035: 2830000, 2036: 2860000, 2037: 2889000, 2038: 2917000, 2039: 2944000,
    2040: 2969000, 2041: 2994000, 2042: 3017000, 2043: 3039000, 2044: 3061000,
    2045: 3081000
}

# GDP per capita projections
gdp_forecast = {
    2025: 49586, 2026: 50131, 2027: 50783, 2028: 51596, 2029: 52421,
    2030: 53207, 2031: 53952, 2032: 54708, 2033: 55474, 2034: 56250,
    2035: 56981, 2036: 57722, 2037: 58473, 2038: 59233, 2039: 60003,
    2040: 60783, 2041: 61512, 2042: 62250, 2043: 63060, 2044: 63879,
    2045: 64710
}

future_df = pd.DataFrame({'years': range(2025, 2046)})
future_df['clark-county_population'] = future_df['years'].map(pop_forecast)
future_df['real_gdp_per_person'] = future_df['years'].map(gdp_forecast)

print(f"    Population: {future_df['clark-county_population'].min():,.0f} → {future_df['clark-county_population'].max():,.0f}")
print(f"    GDP/capita: ${future_df['real_gdp_per_person'].min():,.0f} → ${future_df['real_gdp_per_person'].max():,.0f}")

# Project visitor volumes based on historical growth
# (ML models need this even though we don't have long-term projections)
if 'clark_county_visitors_volumnes' in df_train.columns:
    visitor_growth_rate = df_train['clark_county_visitors_volumnes'].pct_change().mean()
    last_visitor_count = df_train.loc[df_train['years'] == 2019, 'clark_county_visitors_volumnes'].values[0]
    
    # Simple projection: keep growing at historical rate
    future_visitors = []
    current = last_visitor_count
    for _ in range(len(future_df)):
        current *= (1 + visitor_growth_rate)
        future_visitors.append(current)
    
    future_df['clark_county_visitors_volumnes'] = future_visitors
    print(f"    Visitor volume growth rate: {visitor_growth_rate*100:.2f}%/year")

# Store all our forecasts here
all_forecasts = {}

# ============================================================================
# PART 1: MACHINE LEARNING MODELS
# ============================================================================
print("\n[3] Training ML models...")
print("    (These need normalized features)")

# Get features and target
X_ml = df_train[['real_gdp_per_person', 'clark-county_population', 'clark_county_visitors_volumnes']]
y_ml = df_train['enplanments']

# Normalize the target (makes training easier)
y_min, y_max = y_ml.min(), y_ml.max()
y_normalized = (y_ml - y_min) / (y_max - y_min)

# Scale features (RobustScaler handles outliers better)
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X_ml)

# Prep future data
X_future = future_df[['real_gdp_per_person', 'clark-county_population', 'clark_county_visitors_volumnes']]
X_future_scaled = scaler.transform(X_future)

# Define our ML models - tried different configs, these seem reasonable
models_ml = {
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10),
    'Linear Regression (ML)': LinearRegression(),
    'SVR': SVR(kernel='rbf', C=100, gamma='scale'),
    'Decision Tree': DecisionTreeRegressor(random_state=42, max_depth=8)
}

ml_results = {}

for name, model in models_ml.items():
    print(f"\n    Training {name}...")
    model.fit(X_scaled, y_normalized)
    
    # Check training performance
    train_pred = model.predict(X_scaled)
    r2 = r2_score(y_normalized, train_pred)
    rmse = np.sqrt(mean_squared_error(y_normalized, train_pred))
    
    ml_results[name] = {'R²': r2, 'RMSE': rmse}
    
    # Make future predictions
    future_pred_norm = model.predict(X_future_scaled)
    future_pred = future_pred_norm * (y_max - y_min) + y_min  # denormalize
    
    all_forecasts[name] = future_pred
    print(f"      Training R²: {r2:.4f}")
    print(f"      2025: {future_pred[0]:,.0f}  →  2045: {future_pred[-1]:,.0f}")

# ============================================================================
# PART 2: ECONOMETRIC MODELS
# ============================================================================
print("\n[4] Training econometric models...")
print("    (These use log transformations for elasticity)")

# Prep data with log transforms
econ_df = df_train.copy()
for col in ['enplanments', 'clark-county_population', 'real_gdp_per_person']:
    econ_df[col] = econ_df[col].astype(float)

econ_df['log_enplanements'] = np.log(econ_df['enplanments'])
econ_df['log_pop'] = np.log(econ_df['clark-county_population'])
econ_df['log_gdp'] = np.log(econ_df['real_gdp_per_person'])

# Prep future data
future_econ = future_df.copy()
future_econ['log_pop'] = np.log(future_econ['clark-county_population'])
future_econ['log_gdp'] = np.log(future_econ['real_gdp_per_person'])

econ_results = {}

# Model 1: Just population
print(f"\n    OLS (Population only)...")
X_pop = econ_df[['log_pop']].values
y_log = econ_df['log_enplanements'].values

model_pop = LinearRegression()
model_pop.fit(X_pop, y_log)

train_pred = model_pop.predict(X_pop)
r2 = r2_score(y_log, train_pred)
rmse = np.sqrt(mean_squared_error(y_log, train_pred))

econ_results['OLS (Population)'] = {
    'R²': r2, 'RMSE': rmse, 
    'Elasticity': model_pop.coef_[0]
}

future_log = model_pop.predict(future_econ[['log_pop']].values)
all_forecasts['OLS (Population)'] = np.exp(future_log)

print(f"      R²: {r2:.4f}, Elasticity: {model_pop.coef_[0]:.2f}")

# Model 2: Just GDP
print(f"\n    OLS (GDP per capita only)...")
X_gdp = econ_df[['log_gdp']].values

model_gdp = LinearRegression()
model_gdp.fit(X_gdp, y_log)

train_pred = model_gdp.predict(X_gdp)
r2 = r2_score(y_log, train_pred)
rmse = np.sqrt(mean_squared_error(y_log, train_pred))

econ_results['OLS (GDP)'] = {
    'R²': r2, 'RMSE': rmse,
    'Elasticity': model_gdp.coef_[0]
}

future_log = model_gdp.predict(future_econ[['log_gdp']].values)
all_forecasts['OLS (GDP)'] = np.exp(future_log)

print(f"      R²: {r2:.4f}, Elasticity: {model_gdp.coef_[0]:.2f}")

# Model 3: Both predictors
print(f"\n    OLS (Population + GDP)...")
X_both = econ_df[['log_pop', 'log_gdp']].values

model_both = LinearRegression()
model_both.fit(X_both, y_log)

train_pred = model_both.predict(X_both)
r2 = r2_score(y_log, train_pred)
rmse = np.sqrt(mean_squared_error(y_log, train_pred))

econ_results['OLS (Pop + GDP)'] = {
    'R²': r2, 'RMSE': rmse,
    'Pop Elasticity': model_both.coef_[0],
    'GDP Elasticity': model_both.coef_[1]
}

future_log = model_both.predict(future_econ[['log_pop', 'log_gdp']].values)
all_forecasts['OLS (Pop + GDP)'] = np.exp(future_log)

print(f"      R²: {r2:.4f}")
print(f"      Pop elasticity: {model_both.coef_[0]:.2f}, GDP elasticity: {model_both.coef_[1]:.2f}")

# Model 4: Polynomial (degree 3)
print(f"\n    Polynomial regression (degree 3)...")

X_poly_raw = econ_df[['clark-county_population', 'real_gdp_per_person']].values

poly_features = PolynomialFeatures(degree=3, include_bias=False)
X_poly = poly_features.fit_transform(X_poly_raw)

model_poly = LinearRegression()
model_poly.fit(X_poly, y_log)

train_pred = model_poly.predict(X_poly)
r2 = r2_score(y_log, train_pred)
rmse = np.sqrt(mean_squared_error(y_log, train_pred))

econ_results['Polynomial (D=3)'] = {'R²': r2, 'RMSE': rmse}

X_future_raw = future_econ[['clark-county_population', 'real_gdp_per_person']].values
X_future_poly = poly_features.transform(X_future_raw)
future_log = model_poly.predict(X_future_poly)
all_forecasts['Polynomial (D=3)'] = np.exp(future_log)

print(f"      R²: {r2:.4f}")

# ============================================================================
# SAVE RESULTS
# ============================================================================
print("\n[5] Saving results...")

# Create forecast table
forecast_table = pd.DataFrame({'Year': future_df['years']})
for model_name, predictions in all_forecasts.items():
    forecast_table[model_name] = predictions

forecast_table.to_csv('all_models_forecasts_2025_2045.csv', index=False)
print("    ✓ Forecasts saved to CSV")

# Summary stats
summary = pd.DataFrame({
    'Model': list(all_forecasts.keys()),
    '2025': [all_forecasts[m][0] for m in all_forecasts.keys()],
    '2045': [all_forecasts[m][-1] for m in all_forecasts.keys()],
    'Total Growth': [all_forecasts[m][-1] - all_forecasts[m][0] for m in all_forecasts.keys()],
    'Growth %': [(all_forecasts[m][-1] / all_forecasts[m][0] - 1) * 100 for m in all_forecasts.keys()]
})

summary.to_csv('forecast_summary.csv', index=False)

# Training metrics
metrics = []
for name, res in ml_results.items():
    metrics.append({
        'Model': name, 'Type': 'ML',
        'R²': res['R²'], 'RMSE': res['RMSE']
    })
for name, res in econ_results.items():
    row = {'Model': name, 'Type': 'Econometric', 'R²': res['R²'], 'RMSE': res['RMSE']}
    if 'Elasticity' in res:
        row['Elasticity'] = res['Elasticity']
    if 'GDP Elasticity' in res:
        row['GDP_Elasticity'] = res['GDP Elasticity']
    metrics.append(row)

metrics_df = pd.DataFrame(metrics)
metrics_df.to_csv('model_metrics.csv', index=False)

print("    ✓ Summary stats saved")
print("    ✓ Model metrics saved")

# ============================================================================
# VISUALIZATIONS
# ============================================================================
print("\n[6] Creating visualizations...")

# Color schemes (picked these to be distinguishable)
ml_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
econ_colors = ['#95E1D3', '#F38181', '#AA96DA', '#FCBAD3']

# Chart 1: Everything together
print("    Creating combined chart...")
plt.figure(figsize=(15, 9))

plt.plot(df['years'], df['enplanments'], 'o-', 
         color='black', linewidth=2.5, label='Historical', markersize=6)

# ML models (dashed lines)
for i, (name, pred) in enumerate(list(all_forecasts.items())[:4]):
    plt.plot(future_df['years'], pred, 's--', 
             color=ml_colors[i], linewidth=2, label=name, alpha=0.8)

# Econometric models (dash-dot lines)
for i, (name, pred) in enumerate(list(all_forecasts.items())[4:]):
    plt.plot(future_df['years'], pred, 'o-.', 
             color=econ_colors[i], linewidth=2, label=name, alpha=0.8)

plt.axvline(x=2024, color='red', linestyle='--', linewidth=2, alpha=0.7)
plt.xlabel('Year', fontsize=13, fontweight='bold')
plt.ylabel('Enplanements', fontsize=13, fontweight='bold')
plt.title('All 8 Models: 2025-2045 Forecasts\nHarry Reid International Airport', 
          fontsize=15, fontweight='bold')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('all_models_combined.png', dpi=300, bbox_inches='tight')
plt.close()

# Chart 2: ML only
print("    Creating ML-only chart...")
plt.figure(figsize=(13, 7))

plt.plot(df['years'], df['enplanments'], 'o-', 
         color='black', linewidth=2.5, label='Historical', markersize=6)

for i, (name, pred) in enumerate(list(all_forecasts.items())[:4]):
    plt.plot(future_df['years'], pred, 's--', 
             color=ml_colors[i], linewidth=2.5, label=name, alpha=0.8, markersize=5)

plt.axvline(x=2024, color='red', linestyle='--', linewidth=2)
plt.xlabel('Year', fontsize=13, fontweight='bold')
plt.ylabel('Enplanements', fontsize=13, fontweight='bold')
plt.title('Machine Learning Models Only', fontsize=15, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('ml_models_only.png', dpi=300, bbox_inches='tight')
plt.close()

# Chart 3: Econometric only
print("    Creating econometric-only chart...")
plt.figure(figsize=(13, 7))

plt.plot(df['years'], df['enplanments'], 'o-', 
         color='black', linewidth=2.5, label='Historical', markersize=6)

for i, (name, pred) in enumerate(list(all_forecasts.items())[4:]):
    plt.plot(future_df['years'], pred, 'o-.', 
             color=econ_colors[i], linewidth=2.5, label=name, alpha=0.8, markersize=5)

plt.axvline(x=2024, color='red', linestyle='--', linewidth=2)
plt.xlabel('Year', fontsize=13, fontweight='bold')
plt.ylabel('Enplanements', fontsize=13, fontweight='bold')
plt.title('Econometric Models Only', fontsize=15, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('econometric_models_only.png', dpi=300, bbox_inches='tight')
plt.close()

# Chart 4: Uncertainty band
print("    Creating uncertainty visualization...")
plt.figure(figsize=(13, 7))

plt.plot(df['years'], df['enplanments'], 'o-', 
         color='black', linewidth=2.5, label='Historical', markersize=6)

# Calculate forecast range
all_preds = np.array([all_forecasts[m] for m in all_forecasts.keys()])
pred_min = np.min(all_preds, axis=0)
pred_max = np.max(all_preds, axis=0)
pred_mean = np.mean(all_preds, axis=0)

plt.fill_between(future_df['years'], pred_min, pred_max, 
                 alpha=0.2, color='blue', label='Full Range')
plt.plot(future_df['years'], pred_mean, '--', 
         color='blue', linewidth=3, label='Mean', marker='o', markersize=5)

plt.axvline(x=2024, color='red', linestyle='--', linewidth=2)
plt.xlabel('Year', fontsize=13, fontweight='bold')
plt.ylabel('Enplanements', fontsize=13, fontweight='bold')
plt.title('Forecast Uncertainty Across All Models', fontsize=15, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('forecast_uncertainty.png', dpi=300, bbox_inches='tight')
plt.close()

# Chart 5: Individual panels
print("    Creating individual model panels...")
fig, axes = plt.subplots(4, 2, figsize=(16, 18))
axes = axes.flatten()
all_colors = ml_colors + econ_colors

for idx, (name, pred) in enumerate(all_forecasts.items()):
    ax = axes[idx]
    ax.plot(df['years'], df['enplanments'], 'o-', 
            color='black', linewidth=2, label='Historical', markersize=5)
    ax.plot(future_df['years'], pred, 's--', 
            color=all_colors[idx], linewidth=2.5, label='Forecast', markersize=5)
    ax.axvline(x=2024, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.set_xlabel('Year', fontsize=10, fontweight='bold')
    ax.set_ylabel('Enplanements', fontsize=10, fontweight='bold')
    ax.set_title(f'{name}\n2025: {pred[0]/1e6:.1f}M → 2045: {pred[-1]/1e6:.1f}M', 
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.suptitle('Individual Model Forecasts (All 8 Models)', 
             fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('individual_models.png', dpi=300, bbox_inches='tight')
plt.close()

print("    ✓ All charts saved")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*70)
print("SUMMARY")
print("="*70)

forecasts_2045 = {name: pred[-1] for name, pred in all_forecasts.items()}
highest = max(forecasts_2045.items(), key=lambda x: x[1])
lowest = min(forecasts_2045.items(), key=lambda x: x[1])

print(f"\n2045 Forecast Range:")
print(f"  Highest: {highest[0]} = {highest[1]:,.0f}")
print(f"  Lowest:  {lowest[0]} = {lowest[1]:,.0f}")
print(f"  Spread:  {highest[1] - lowest[1]:,.0f} passengers")
print(f"\nMean forecast: {pred_mean[-1]:,.0f}")

print("\nGrowth rates (CAGR):")
for name in all_forecasts.keys():
    start = all_forecasts[name][0]
    end = all_forecasts[name][-1]
    cagr = ((end / start) ** (1/20) - 1) * 100
    print(f"  {name:25s} {cagr:>6.2f}%")

print("\n" + "="*70)
print("DONE! Files generated:")
print("  - all_models_forecasts_2025_2045.csv")
print("  - forecast_summary.csv")
print("  - model_metrics.csv")
print("  - all_models_combined.png")
print("  - ml_models_only.png")
print("  - econometric_models_only.png")
print("  - forecast_uncertainty.png")
print("  - individual_models.png")
print("="*70)
