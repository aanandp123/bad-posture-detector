import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def analyze_posture(results):
    # Create a Pandas df from the results
    df_results = pd.DataFrame(results)

    # Calculate the percentage of time in each posture
    percentage_good_posture = (df_results['Posture'].sum() / len(df_results)) * 100
    percentage_bad_posture = 100 - percentage_good_posture

    # Display the percentage of time in each posture
    print(f"Percentage of time in Good Posture: {percentage_good_posture:.2f}%")
    print(f"Percentage of time in Bad Posture: {percentage_bad_posture:.2f}%")

    # Creat Seaborn chart
    sns.set(style="whitegrid")

    # Chart 1: Percentage of time in each posture
    sns.barplot(x=['Good Posture', 'Bad Posture'], y=[percentage_good_posture, percentage_bad_posture])
    plt.title('Percentage of Time in Good and Bad Posture')
    plt.xlabel('Posture')
    plt.ylabel('Percentage')
    plt.show()

    sns.set_style("whitegrid")
    # Chart 2: Line graph showing time periods when it is good and bad
    sns.lineplot(x='Timestamp', y='Posture', data=df_results)
    plt.title('Posture Over Time')
    plt.xlabel('Timestamp')
    plt.ylabel('Posture (0 - Bad, 1 - Good)')
    plt.show()


