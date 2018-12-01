import datalabels from 'chartjs-plugin-datalabels';
import { computed } from "mobx";
import { observer } from "mobx-react";
import * as React from 'react';
import { HorizontalBar } from "react-chartjs-2";
import { createDataSets, formatTime, RANGES } from "../../data";
import CharStore from '../../stores/ChartStore';
import TraceStore from "../../stores/TraceStore";
// import { repeat } from "../../utils";
import './index.less';

@observer
class TraceBody extends React.Component {
  public render() {
    return (
      <div className="trace-body">
        <div className="trace-body-content">
          <div className="trace-body-side">
            <header className="trace-body-side-header" />
            <ul className="trace-body-side-list">
              <li>TiDB GetTSAsync</li>
              <li>TiDB pdclient.processTSORequests</li>
              <li>TiDB session.ParseSQL</li>
              <li>TiDB executor.Compile</li>
            </ul>
          </div>
          <div className="trace-body-graph">
            <HorizontalBar
              height={60}
              // tslint:disable
              datasetKeyProvider={(e: any) => e.key}
              data={{
                labels: this.spanList.map(span => span.operationName),
                datasets: createDataSets(this.spanList),
              }}
              options={{
                legend: {
                  display: false,
                },
                tooltips: {
                  enabled: false,
                  custom(this: any, tooltipModel: any) {
                    let dataPoint = {};
                    if (tooltipModel.dataPoints && tooltipModel.dataPoints.length) {
                      dataPoint = RANGES[tooltipModel.dataPoints[0].index] || {};
                    }
                    // Tooltip Element
                    let tooltipEl = document.getElementById('chartjs-tooltip');

                    // Create element on first render
                    if (!tooltipEl) {
                      tooltipEl = document.createElement('div');
                      tooltipEl.id = 'chartjs-tooltip';
                      document.body.appendChild(tooltipEl);
                    }

                    // Hide if no tooltip
                    if (tooltipModel.opacity === 0) {
                      (tooltipEl.style as any).opacity = 0;
                      return;
                    }

                    // Set caret Position
                    tooltipEl.classList.remove('above', 'below', 'no-transform');
                    if (tooltipModel.yAlign) {
                      tooltipEl.classList.add(tooltipModel.yAlign);
                    } else {
                      tooltipEl.classList.add('no-transform');
                    }

                    tooltipEl.innerHTML =
                      `<div class="trace-body-tooltip">
                    ${JSON.stringify(dataPoint, null, 2)}
                   </div>`;

                    // `this` will be the overall tooltip
                    const position = this._chart.canvas.getBoundingClientRect();

                    // Display, position, and set styles for font
                    (tooltipEl.style as any).opacity = 1;
                    tooltipEl.style.position = 'absolute';
                    tooltipEl.style.left = position.left + window.pageXOffset + tooltipModel.caretX + 'px';
                    tooltipEl.style.top = position.top + window.pageYOffset + tooltipModel.caretY + 'px';
                    tooltipEl.style.fontFamily = tooltipModel._bodyFontFamily;
                    tooltipEl.style.fontSize = tooltipModel.bodyFontSize + 'px';
                    tooltipEl.style.fontStyle = tooltipModel._bodyFontStyle;
                    tooltipEl.style.padding = tooltipModel.yPadding + 'px ' + tooltipModel.xPadding + 'px';
                    tooltipEl.style.pointerEvents = 'none';
                  }
                },
                scales: {
                  yAxes: [{
                    barPercentage: 1,
                    categoryPercentage: 0.25,
                    stacked: true,
                    gridLines: {
                      // display: false,
                      color: '#777777',
                      // borderDash: [3, 1.5],
                      // zeroLineBorderDash: [3, 1.5],
                    },
                  }],
                  xAxes: [{
                    stacked: true,
                    position: 'top',
                    gridLines: {
                      color: '#777777',
                      // zeroLineWidth: 2,
                    },
                    ticks: {
                      callback: formatTime,
                      min: CharStore.minValue,
                      max: Math.floor(CharStore.maxValue + (CharStore.maxValue - CharStore.minValue) * 0.1),
                    },
                  }],
                },
                plugins: {
                  datalabels: {
                    display: (context: any) => {
                      return context.datasetIndex === 1;
                    },
                    align: 'end',
                    anchor: 'end',
                    color: '#fefefe',
                    formatter: formatTime,
                  }
                },
              } as any}
              plugins={[datalabels]}
            />
          </div>
        </div>
      </div>
    )
  }

  @computed private get spanList() {
    return TraceStore.getSpanList(TraceStore.traceId || '');
  }
}

export default TraceBody;
