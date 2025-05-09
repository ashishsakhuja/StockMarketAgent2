from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from dotenv import load_dotenv
from stockmarketagent2.tools.custom_tool import (
    StockDataTool,
    StockNewsTool,
    ForecastPriceTool,
    EstimateROITool
)

load_dotenv()

# If you want to run a snippet of code before or after the crew starts, 
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators

@CrewBase
class Stockmarketagent2():
	"""Stockmarketagent2 crew"""

	agents_config = 'config/agents.yaml'
	tasks_config = 'config/tasks.yaml'
	tools_config = [
		StockDataTool(),
		StockNewsTool(),
		ForecastPriceTool(),
		EstimateROITool()
	]

	@agent
	def stock_data_analyst(self) -> Agent:
		return Agent(
			config=self.agents_config['stock_data_analyst'],
			verbose=True
		)

	@agent
	def news_analyst(self) -> Agent:
		return Agent(
			config=self.agents_config['news_analyst'],
			verbose=True
		)

	@agent
	def investment_advisor(self) -> Agent:
		return Agent(
			config=self.agents_config['investment_advisor'],
			verbose=True
		)

	@task
	def fetch_stock_data(self) -> Task:
		return Task(
			config=self.tasks_config["fetch_stock_data"],
			agent=self.stock_data_analyst(),
		)

	@task
	def fetch_news(self) -> Task:
		return Task(
			config=self.tasks_config["fetch_news"],
			agent=self.news_analyst(),
		)

	@task
	def forecast_price(self) -> Task:
		return Task(
			config=self.tasks_config["forecast_price"],
			agent=self.investment_advisor(),
		)

	@task
	def estimate_roi(self) -> Task:
		return Task(
			config=self.tasks_config["estimate_roi"],
			agent=self.investment_advisor(),
		)

	@task
	def provide_recommendation(self) -> Task:
		return Task(
			config=self.tasks_config["provide_recommendation"],
			agent=self.investment_advisor(),
		)

	@crew
	def crew(self) -> Crew:
		"""Creates the Stockmarketagent2 crew"""
		return Crew(
			agents=[
				self.stock_data_analyst(),
				self.news_analyst(),
				self.investment_advisor(),
			],
			tasks=[
				self.fetch_stock_data(),
				self.fetch_news(),
				self.forecast_price(),
				self.estimate_roi(),
				self.provide_recommendation(),
			],

			process=Process.sequential,  # or Process.hierarchical
			verbose=True
		)
