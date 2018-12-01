import * as monaco from 'monaco-editor';
import * as React from 'react';
import TomorrowNightTheme from './tomorrow-night';

import './index.less';

monaco.editor.defineTheme('tomorrow-night', TomorrowNightTheme as any);
monaco.editor.setTheme('tomorrow-night');

class Monaco extends React.Component {
  private monacoEditor: HTMLDivElement;

  public componentDidMount() {
    monaco.editor.create(this.monacoEditor, {
      language: 'sql',
      value: `select * from table where column="value";`,
    });
  }

  public render() {
    return (
      <div style={{ paddingTop: 16, backgroundColor: '#1d1e20' }}>
        <div
          className="monaco-editor"
          ref={this.refMonacoEditor}
          style={{ height: 220 }}
        />
      </div>
    );
  }

  private refMonacoEditor = (ref: HTMLDivElement) => {
    this.monacoEditor = ref;
  };
}

export default Monaco;
