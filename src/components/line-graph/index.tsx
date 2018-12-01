import 'chartjs-plugin-streaming';
import * as React from 'react';
import { Line } from 'react-chartjs-2';
import './index.less';

class LineGraph extends React.Component {
  public state = {
    data: [],
  };

  public render() {
    return (
      <div className="line-graph">
        <header className="line-graph-header">
          <select className="value-select" value="cpu">
            <option value="cpu">CPU Usage</option>
          </select>
        </header>
        <Line
          height={80}
          data={{
            datasets: [{
              label: 'CPU',
              data: this.state.data,
              backgroundColor: 'rgba(54, 162, 235, 0.6)',
              borderColor: 'rgba(54, 162, 235, 0.6)',
              fill: false,
              cubicInterpolationMode: 'monotone',
            }],
          }}
          options={{
            scales: {
              xAxes: [{
                type: 'realtime',
                realtime: {
                  duration: 20000,
                  refresh: 1000,
                  delay: 2000,
                  onRefresh: (chart: any) => {
                    chart.data.datasets.forEach((dataset: any) => {
                      dataset.data.push({
                        x: Date.now(),
                        y: Math.floor(Math.random() * 100),
                      });
                    });
                  },
                },
                ticks: {
                  min: 0,
                  max: 100,
                },
              } as any],
              yAxes: {
                ticks: {
                  min: 0,
                  max: 100,
                },
              } as any,
            },
            tooltips: {
              mode: 'nearest',
              intersect: false
            },
            hover: {
              mode: 'nearest',
              intersect: false
            },
            plugins: {
              streaming: {
                frameRate: 30,
              },
              datalabels: {
                display: false,
              },
            },
          }}
        />
      </div>
    );
  }
}

export default LineGraph;
