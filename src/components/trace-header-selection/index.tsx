import { autorun, computed, IReactionDisposer } from "mobx";
import { observer } from "mobx-react";
import * as React from 'react';
import { HorizontalBar } from 'react-chartjs-2';
import { createDataSets } from "../../data";
import CharStore from '../../stores/ChartStore';
import TraceStore from "../../stores/TraceStore";
// import { repeat } from "../../utils";
import './index.less';

interface IProps {
  startPercent?: number;
  endPercent?: number;
}

interface IState {
  startPercent: number;
  endPercent: number;
  startX: number;
  endX: number;
}

const ignore = (e: any) => {
  e.preventDefault();
};

const ignoreMouseEvent = {
  onMouseDown: ignore,
  onMouseMove: ignore,
  onMouseUp: ignore,
};

@observer
class TraceHeaderSelection extends React.Component<IProps, IState> {
  public state: IState = {
    startPercent: 0,
    endPercent: 100,
    startX: 0,
    endX: 0,
  };

  private isSelecting: boolean = false;
  private dispose: IReactionDisposer;

  public componentDidMount() {
    this.dispose = autorun(() => {
      CharStore.setMinValue(this.minValue);
      CharStore.setMaxValue(this.maxValue);
    })
  }

  public componentWillReceiveProps(nextProps: IProps) {
    this.setState({
      startPercent: nextProps.startPercent || 0,
      endPercent: nextProps.endPercent || 100,
    });
  }

  public componentWillUnmount() {
    this.dispose();
  }

  public render() {
    let { startPercent, endPercent } = this.state;
    const { startX, endX } = this.state;
    if (startPercent > endPercent) {
      [endPercent, startPercent] = [startPercent, endPercent];
    }
    const MASK_COLOR = 'rgba(255,255,255,0.25)';
    const PLAIN_COLOR = 'rgba(0,0,0,0)';
    const progress = `linear-gradient(to right, ${
      `${MASK_COLOR} 0%`}, ${
      `${MASK_COLOR} ${startPercent || 0}%`}, ${
      `${PLAIN_COLOR} ${startPercent || 0}%`}, ${
      `${PLAIN_COLOR} ${endPercent || 100}%`}, ${
      `${MASK_COLOR} ${endPercent || 100}%`}, ${
      `${MASK_COLOR} 100%`})`;
    return (
      <>
        <header className="trace-header">Traces</header>
        <div className="trace-header-selection">
          <HorizontalBar
            height={10}
            // tslint:disable
            datasetKeyProvider={(e: any) => e.key}
            data={{
              labels: this.spanList.map(() => ''),
              datasets: this.datesets,
            }}
            options={{
              legend: {
                display: false,
              },
              tooltips: {
                enabled: false,
              },
              scales: {
                scaleLabel: {
                  display: false,
                },
                yAxes: [{
                  barPercentage: 1,
                  categoryPercentage: 1,
                  stacked: true,
                  gridLines: {
                    display: false,
                  },
                }],
                xAxes: [{
                  stacked: true,
                  position: 'top',
                  gridLines: {
                    color: '#777777',
                  },
                  ticks: {
                    callback(value: number) {
                      return `${value}ms`;
                    },
                    min: this.minValue,
                    max: this.maxValue,
                  },
                }],
              },
              plugins: {
                datalabels: {
                  display: false,
                },
              },
            } as any}
          />
          <div
            className="trace-header-mask"
            style={{
              backgroundImage: progress,
            }}
            // tslint:disable
            onMouseDown={this.startSelectRange}
            onMouseMove={this.moveSelectRange}
            onMouseUp={this.endSelectRange}
          >
            <div
              className="split-start"
              style={{ transform: `translateX(${startX - 3}px)` }}
              {...ignoreMouseEvent}
            />
            <div
              className="split-end"
              style={{ transform: `translateX(${endX - 3}px)` }}
              {...ignoreMouseEvent}
            />
          </div>
        </div>
      </>
    );
  }

  @computed
  private get datesets() {
    return createDataSets(this.spanList);
  }

  @computed
  private get spanList() {
    return TraceStore.getSpanList(TraceStore.traceId || '');
  }

  private get minValue() {
    return 0;
  }

  @computed
  private get maxValue() {
    const minimum = Math.min(...this.spanList.map(r => r.startTime)) || 0;
    const maximum = Math.max(...this.spanList.map(r => r.startTime + r.duration)) || 0;
    return maximum - minimum;
  }

  private getPercentage(e: React.MouseEvent) {
    const rect = (e.currentTarget as HTMLDivElement).getBoundingClientRect();
    return 100 * (e.pageX - rect.left) / rect.width;
  }

  private startSelectRange = (e: React.MouseEvent) => {
    if (!e.target || this.isSelecting) {
      return;
    }
    this.isSelecting = true;
    const percentage = this.getPercentage(e);
    const rect = (e.currentTarget as HTMLDivElement).getBoundingClientRect();
    const clientX = e.pageX - rect.left;
    this.setState({
      startPercent: percentage,
      endPercent: percentage,
      startX: clientX,
      endX: clientX,
    });
  };

  private moveSelectRange = (e: React.MouseEvent) => {
    if (!e.target || !this.isSelecting) {
      return;
    }
    const percentage = this.getPercentage(e);
    const rect = (e.currentTarget as HTMLDivElement).getBoundingClientRect();
    const clientX = e.pageX - rect.left;
    this.setState({
      endPercent: percentage,
      endX: clientX,
    });
  };

  private endSelectRange = (e: React.MouseEvent) => {
    if (!e.target) {
      return;
    }
    this.isSelecting = false;
    const percentage = this.getPercentage(e);
    const rect = (e.currentTarget as HTMLDivElement).getBoundingClientRect();
    const clientX = e.pageX - rect.left;
    this.setState({
      endPercent: percentage,
      endX: clientX,
    });
    CharStore.setMinValue(+(this.minValue + (this.maxValue - this.minValue) * this.state.startPercent / 100).toFixed(1));
    CharStore.setMaxValue(+(this.minValue + (this.maxValue - this.minValue) * percentage / 100).toFixed(1));
  };
}

export default TraceHeaderSelection;
