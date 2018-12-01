import { action, computed, observable } from "mobx";
import { ISpan, ITrace } from "../data";
import request from '../request';

interface ILevelSpan extends ISpan {
  level: number;
}

class TraceStore {
  @observable public traces = new Map<string, ITrace>();
  @observable public spans = new Map<string, ISpan>();
  @observable public expandedSpans = new Map<string, boolean>();

  @computed get trace(): ITrace | null {
    const item = this.traces.values().next();
    if (item.done) {
      return null;
    }
    return item.value;
  }

  @computed get traceId(): string | null {
    return this.trace ? this.trace.traceID : null;
  }

  public getTopSpans(traceId: string): ISpan[] {
    const trace = this.traces.get(traceId);
    return trace ? trace.spans.filter(span => span.references.length === 0) : [];
  }

  public getChildSpans(spanId: string): ISpan[] {
    const spans: ISpan[] = [];
    for (const span of this.spans.values()) {
      if (span.references.some(r => r.spanID === spanId)) {
        spans.push(span);
      }
    }
    return spans;
  }

  public getSpanList(traceId: string): ILevelSpan[] {
    const spanList: ILevelSpan[] = [];
    for (const span of this.getTopSpans(traceId)) {
      this.appendSpanList(spanList, span, 0);
    }
    return spanList;
  }

  @action public loadTraces() {
    request.get('/api/traces', {
      params: {
        end: +new Date() * 1000,
        limit: 20,
        lookback: '1h',
        service: 'TiDB',
        start: (+new Date() - 3600 * 1000) * 1000,
      },
    }).then((res) => {
      for (const trace of res.data.data) {
        this.addTrace(trace);
        for (const span of trace.spans) {
          this.addSpan(span);
        }
      }
    });
  }

  private appendSpanList(list: ILevelSpan[], span: ISpan, level: number) {
    list.push({
      ...span,
      level,
    });
    for (const child of this.getChildSpans(span.spanID)) {
      this.appendSpanList(list, child, level + 1);
    }
  }


  @action private addTrace(trace: ITrace) {
    this.traces.set(trace.traceID, trace);
  }

  @action private addSpan(span: ISpan) {
    this.spans.set(span.spanID, span);
  }
}

export default new TraceStore();
