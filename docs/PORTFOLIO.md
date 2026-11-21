# RouteWise AI: portfolio content

Live app: https://routewise-ai-zubairlabs.streamlit.app
Code: https://github.com/muhammadzubair-data/routewise-ai-supply-chain-intelligence

All numbers below come from the synthetic dataset in the repository (6,000 shipments, seed 42). Keep the "synthetic data" note wherever you reuse them.

---

## 1. Case study

### The logistics problem

Most operations teams find out a shipment is late once it is already late. At that point every option costs more: expediting, rebooking, or explaining a missed delivery to a customer. The useful question is earlier and simpler: of the shipments leaving today, which ones are most likely to slip, and what is driving it?

### Objectives

- Estimate delay risk at the moment of dispatch, using only information available then.
- Turn the estimates into something a planner can act on: a ranked list, clear risk bands, and a way to test changes before making them.
- Evaluate the model the way it would actually be used, on future shipments it has never seen.
- Be transparent about what the data is and what the results do and do not show.

### Data engineering

The project uses a seeded synthetic generator so every result can be reproduced. It simulates a year of shipments across four transport modes, five regions and three service levels. Weather forecasts get worse in winter and warehouse backlog builds up in November and December, so the last quarter is harder than the rest of the year, as it would be in real life.

Each record is split into two groups: what is known at dispatch (distance, weight, mode, forecasts, carrier history, whether customs is needed) and what is only known later (whether it was delayed, by how much, and how long it sat in customs). Only the first group ever reaches the model. A validation step drops rows with unknown categories or impossible values and reports why, while keeping missing values for the pipeline to fill.

### Machine-learning approach

- Temporal split: train on January to September, test on October to December.
- Three candidate models compared with 5-fold cross-validation on the training months only. Logistic Regression won on PR-AUC and gave the best-calibrated probabilities.
- The decision threshold was tuned on out-of-fold training predictions, so the test quarter was never used to make a choice.
- The dashboard never shows a score from a model that has seen that shipment.

One real fix came out of the audit: the original version used customs hold hours as an input. That number only exists after the shipment reaches customs, so it was leaking the outcome. It was replaced with "customs clearance required", which is known at booking. The app shows the difference openly.

### Dashboard

Six pages in a multipage Streamlit app with light and dark themes:

- Command Center: network KPIs, monthly trend with the test period marked, risk mix, and delay drivers.
- Delay Prediction: score a single shipment and see what is pushing its risk up or down.
- Risk Explorer: filter, weight by service level, and export a ranked work list.
- What-If Simulator: change carrier, mode, backlog or customs and compare against the baseline.
- Model Performance: comparison, curves, confusion matrices, calibration, importance, and the leakage check.
- Data Explorer: feature availability, data quality, distributions and downloads.

### Results (synthetic data)

On 1,500 shipments from the unseen final quarter:

- ROC-AUC 0.778, PR-AUC 0.531 against a 0.233 baseline.
- At the tuned threshold, the model caught 71% of late shipments, with about 1.4 false alarms for every correct flag.
- Predicted and observed delay rates lined up closely in every risk band: 8.6% vs 9.1% (Low), 30.4% vs 30.6% (Medium), 59.4% vs 59.1% (High).
- The four strongest signals were traffic and weather forecasts, carrier on-time history and warehouse backlog. Inputs that have no effect in the simulation (weight, planned days, region) came out near zero, which is a good sign the model is not inventing patterns.

These figures describe how well the model recovers a simulated pattern. They are not a claim about real deliveries or cost savings.

### Where this applies

- Morning dispatch review: a short list of shipments to watch or expedite.
- Carrier conversations: show how much carrier reliability moves risk.
- Peak-season planning: see how backlog and weather push the network.
- A template for teams who want to build the same thing on their own shipment history.

### Limitations and next steps

Synthetic data only, one test quarter, idealised forecasts, and no live integrations. Next steps: validate on real licensed data, run rolling back-tests, compare a gradient-boosted model, choose thresholds from actual false-alarm and missed-delay costs, and add drift monitoring.

---

## 2. LinkedIn launch post

Which of today's shipments are going to be late?

That is the question I built RouteWise AI around. It is a delivery-delay prediction and risk dashboard that only uses information you actually have at dispatch: booking details, weather and traffic forecasts, carrier history and warehouse backlog.

A few things I cared about while building it:

→ Tested on the future, not on a random shuffle. The model trains on January to September and is scored on October to December, including the peak season.
→ No leakage. The first version used customs hold hours as an input, which you only know after the fact. I took it out and the app shows exactly how much it would have inflated the score.
→ Probabilities you can trust. In each risk band, predicted and observed delay rates are within about one percentage point.
→ Something a planner can use: a ranked risk list with CSV export, a single-shipment predictor, and a what-if simulator.

On the unseen final quarter it reaches ROC-AUC 0.778 and catches about 7 in 10 late shipments at the tuned threshold.

Important: the data is synthetic and fully reproducible, so these numbers show the method works on a simulated pattern. They are not a real-world accuracy claim.

Live app: https://routewise-ai-zubairlabs.streamlit.app
Code: https://github.com/muhammadzubair-data/routewise-ai-supply-chain-intelligence

If you work in logistics or supply chain, I'd like to hear which input you'd add first.

#DataScience #MachineLearning #SupplyChain #Logistics #Streamlit #Python

---

## 3. Upwork portfolio description

Title: RouteWise AI – Delivery Delay Prediction and Shipment Risk Dashboard

Description:
A machine-learning dashboard that estimates which shipments are likely to arrive late, using only information available at dispatch time.

What I built:
- A Python data pipeline with validation and a clear split between dispatch-time inputs and after-the-fact outcomes
- A delay model chosen by cross-validation and tested on a later, unseen quarter
- A six-page Streamlit app: command center, single-shipment predictor, risk explorer with CSV export, what-if simulator, model report and data explorer
- Light and dark themes, interactive Plotly charts, automated tests and a GitHub Actions pipeline

Results on the synthetic test quarter: ROC-AUC 0.778, about 71% of late shipments caught at the tuned threshold, and predicted vs observed delay rates within about one point in every risk band.

Note: built on reproducible synthetic data. The same approach can be applied to your own shipment history.

Tools: Python, pandas, scikit-learn, Streamlit, Plotly, pytest, Playwright, GitHub Actions

Live demo: https://routewise-ai-zubairlabs.streamlit.app
Code: https://github.com/muhammadzubair-data/routewise-ai-supply-chain-intelligence

Skills: Machine Learning, Data Science, Python, Streamlit, Predictive Modeling, Supply Chain Analytics, Dashboard Development, scikit-learn

---

## 4. Six LinkedIn carousel concepts

**Carousel 1: "Know it's late before it's late"**
Slide 1 hook: "Your late shipments were predictable at dispatch." Slides 2 to 5: what is known at dispatch vs after delivery, the risk bands, the ranked list. Last slide: live demo link.
Visual: Command Center screenshot with the holdout period shaded.

**Carousel 2: "The feature I had to delete"**
Story of the customs-hold leak. Slide by slide: what the feature was, why it looked useful, why it is unknown at dispatch, the score with and without it (PR-AUC 0.558 vs 0.531), the rule: "If you can't know it when you predict, you can't use it."
Visual: the leakage note from Model Performance.

**Carousel 3: "Random split vs time split"**
Why a shuffled test set answers the wrong question for operations. Show the timeline (Jan to Sep train, Oct to Dec test) and the two sets of numbers side by side.
Visual: the data-pipeline illustration and the monthly chart.

**Carousel 4: "0.5 is not a magic number"**
Same model, two thresholds: at 0.50 it catches 28% of delays, at 0.214 it catches 71% with more false alarms. Explain how to pick based on what a missed delay costs.
Visual: the two confusion matrices.

**Carousel 5: "Can you trust a 30% risk score?"**
Calibration explained simply: in the Medium band the model said 30.4% and 30.6% were actually late. Why calibration matters more than accuracy for planning.
Visual: calibration chart and band table.

**Carousel 6: "Five what-if questions for your dispatch desk"**
Use the simulator: switch to a more reliable carrier, clear customs, reduce backlog, change mode, avoid the worst-weather day. One slide per question with before and after estimates (taken live from the app).
Visual: What-If Simulator with the tornado chart.
