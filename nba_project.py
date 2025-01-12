#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 16:38:06 2024

@author: shaiyaan
"""

import pandas as pd
from nba_api.stats.endpoints import playergamelog
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from nba_api.stats.static import players
from nba_api.stats.endpoints import playergamelog, teamgamelog, commonplayerinfo
from nba_api.stats.static import players, teams
import seaborn as sns

def fetch_player_data(player_name, season):
    """
    Fetch player game logs for a specific season.
    """
    player_id = next(p['id'] for p in players.get_players() if p['full_name'].lower() == player_name.lower())
    gamelog = playergamelog.PlayerGameLog(player_id=player_id, season=season)
    return gamelog.get_data_frames()[0]

def fetch_team_stats(team_id, season):
    """
    Fetch team stats for a specific season.
    """
    team_logs = teamgamelog.TeamGameLog(team_id=team_id, season=season)
    return team_logs.get_data_frames()[0]

def get_opponent_team(matchup):
    if '@' in matchup:
        return matchup.split('@')[1].strip() # away game
    elif 'vs.' in matchup:
        return matchup.split('vs.')[1].strip() # home game
    return None

def calculate_features(player_logs, opponent_logs):
    """
    Add calculated features such as eFG%, Usage Rate, and opponent stats.
    """
    # Player-specific stats
    player_logs['eFG%'] = (player_logs['FGM'] + 0.5 * player_logs['FG3M']) / player_logs['FGA']
    player_logs['Usage%'] = (player_logs['FGA'] + 0.44 * player_logs['FTA'] + player_logs['TOV']) / player_logs['MIN']

    # Opponent-specific stats
    opponent_logs['Possessions'] = opponent_logs['FGA'] - opponent_logs['OREB'] + opponent_logs['TOV'] + (0.44 * opponent_logs['FTA'])
    opponent_logs['DEF_RATING'] = (opponent_logs['PTS'] / opponent_logs['Possessions']) * 100

    # Rolling averages for opponent defensive stats
    rolling_stats = ['DEF_RATING', 'REB', 'STL', 'BLK', 'TOV']
    for stat in rolling_stats:
        opponent_logs[f'{stat}_Rolling'] = opponent_logs[stat].rolling(5, min_periods=1).mean()

    player_logs['Opponent_Team'] = player_logs['MATCHUP'].apply(get_opponent_team)
    opponent_logs['Team_Name'] = opponent_logs['MATCHUP'].apply(lambda x: x.split(' ')[0])

    # Merge on GAME_DATE and Opponent_Team
    merged = pd.merge(
        player_logs,
        opponent_logs,
        left_on=['GAME_DATE', 'Opponent_Team'],
        right_on=['GAME_DATE', 'Team_Name'],
        suffixes=('', '_Opponent')
    )

    # Add home and rest days
    merged['Is_Home'] = merged['MATCHUP'].apply(lambda x: 1 if 'vs.' in x else 0)
    merged['Rest_Days'] = merged['GAME_DATE'].diff().dt.days.fillna(0).astype(int)

    return merged

def train_models(X, y):
    """
    Trains Random Forest models for points, rebounds, and assists.
    """
    models = {}
    for target in y.columns:
        X_train, X_test, y_train, y_test = train_test_split(X, y[target], test_size=0.2, random_state=42)
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        print(f'{target} MAE: {mae:.2f}')
        models[target] = model
    return models

def predict_hypothetical_matchup(models, player_logs, opponent_logs, team_name):
    """
    Predict Points, Rebounds, and Assists for a hypothetical next game against a given opponent.
    """
    # Filter opponent logs for the specific team and take the latest row
    opponent_features = opponent_logs[opponent_logs['Team_Name'] == team_name].iloc[-1:]

    # Merge with a recent row of player data to ensure feature alignment
    player_sample = player_logs.iloc[-1:]  # Take the latest player row
    hypothetical_matchup = pd.concat([player_sample.reset_index(drop=True), opponent_features.reset_index(drop=True)], axis=1)

    # Extract features used during model training
    features = hypothetical_matchup[['eFG%', 'Usage%', 'DEF_RATING_Rolling', 'REB_Rolling', 'STL_Rolling', 'BLK_Rolling']]

    # Ensure feature alignment with model
    predictions = {stat: model.predict(features)[0] for stat, model in models.items()}

    # Output results
    print(f"\nHypothetical matchup against {team_name}:")
    print(f"Predicted Points: {predictions['PTS']:.2f}")
    print(f"Predicted Rebounds: {predictions['REB']:.2f}")
    print(f"Predicted Assists: {predictions['AST']:.2f}")

    return predictions


def predict_performance_all(models, X):
    """
    Predict performance (PTS, REB, AST) for all rows in the dataset.
    """
    predictions = pd.DataFrame(index=X.index)  # Create an empty DataFrame with the same index as X

    for stat, model in models.items():
        predictions[stat] = model.predict(X)  # Predict for the entire dataset

    return predictions

# Main script
player_name = input("Please enter a player's name: ").strip()  # Example: "LeBron James"
team_name = input("Enter the team they are playing against: ").strip()  # Example: "Phoenix Suns"

seasons = [f"{year}-{str(year+1)[-2:]}" for year in range(2021, 2025)]  # Dynamically generate seasons
player_logs_list = []
opponent_logs_list = []

for season in seasons:
    player_logs_list.append(fetch_player_data(player_name, season))

player_logs = pd.concat(player_logs_list, ignore_index=True)
player_logs['GAME_DATE'] = pd.to_datetime(player_logs['GAME_DATE'], format='%b %d, %Y')

team_id = next(t['id'] for t in teams.get_teams() if t['full_name'] == team_name)
for season in seasons:
    opponent_logs_list.append(fetch_team_stats(team_id, season))

opponent_logs = pd.concat(opponent_logs_list, ignore_index=True)
opponent_logs['GAME_DATE'] = pd.to_datetime(opponent_logs['GAME_DATE'], format='%b %d, %Y')

merge = calculate_features(player_logs, opponent_logs)
features = merge[['eFG%', 'Usage%', 'DEF_RATING_Rolling', 'REB_Rolling', 'STL_Rolling', 'BLK_Rolling']]
target = merge[['PTS', 'REB', 'AST']]

models = train_models(features, target)
predicted_stats = predict_performance_all(models, features)

predictions = predict_hypothetical_matchup(models, player_logs, opponent_logs, team_name)

comparison = pd.concat([target, predicted_stats], axis=1)
comparison.columns = ['Actual_PTS', 'Actual_REB', 'Actual_AST', 'Predicted_PTS', 'Predicted_REB', 'Predicted_AST']

for_me = comparison[['Actual_PTS', 'Predicted_PTS','Actual_AST', 'Predicted_AST','Actual_REB', 'Predicted_REB']].copy()
for_me['GAME_DATE'] = merge['GAME_DATE']
for_me = for_me.sort_values(by='GAME_DATE')
print(for_me)

import matplotlib.pyplot as plt

# Correlation Heatmap
correlation_matrix = merge[['PTS', 'eFG%', 'Usage%', 'DEF_RATING_Rolling', 'REB_Rolling']].corr()

# Plot Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title("Correlation Between Features and Player Points")
plt.show()

# Plot points
plt.figure(figsize=(8, 6))
plt.scatter(comparison['Actual_PTS'], comparison['Predicted_PTS'], alpha=0.7)
plt.plot([0, max(comparison['Actual_PTS'])], [0, max(comparison['Actual_PTS'])], color='green', linestyle='--')
plt.title("Actual vs. Predicted Points")
plt.xlabel("Actual Points")
plt.ylabel("Predicted Points")
plt.show()



# Display predictions
#print(predicted_stats)
#

#  Weighted averaging
# long_term_stats = dataset.query("Season == '2023-24'")
# short_term_stats = dataset.query("Season == '2024-25'")
# dataset['Blended_PTS'] = 0.8 * long_term_stats['PTS'] + 0.2 * short_term_stats['PTS']

# Save final dataset