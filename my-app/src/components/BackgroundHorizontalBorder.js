import { useMemo, useCallback } from "react";
import { useNavigate } from "react-router-dom";
import PropTypes from "prop-types";
import styles from "./BackgroundHorizontalBorder.module.css";

const BackgroundHorizontalBorder = ({
  className = "",
  mainLogob9ffbb6svg,
  linkWidth,
  aColor,
  containerColor,
  buttonWidth,
  onLinkContainerClick,
  linkBoxShadow,
  linkBackgroundColor,
  linkBorder,
}) => {
  const mainLogob9ffbb6svgIconStyle = useMemo(() => {
    return {
      width: linkWidth,
    };
  }, [linkWidth]);

  const aStyle = useMemo(() => {
    return {
      color: aColor,
    };
  }, [aColor]);

  const a1Style = useMemo(() => {
    return {
      color: containerColor,
    };
  }, [containerColor]);

  const rectangleStyle = useMemo(() => {
    return {
      width: buttonWidth,
    };
  }, [buttonWidth]);

  const linkStyle = useMemo(() => {
    return {
      boxShadow: linkBoxShadow,
      backgroundColor: linkBackgroundColor,
      border: linkBorder,
    };
  }, [linkBoxShadow, linkBackgroundColor, linkBorder]);

  const navigate = useNavigate();

  const onLinkContainerClick1 = useCallback(() => {
    navigate("/");
  }, [navigate]);

  return (
    <header
      className={[styles.backgroundhorizontalborder, className].join(" ")}
    >
      <div className={styles.container}>
        <div className={styles.frameParent}>
          <div className={styles.linkWrapper}>
            <div
              className={styles.link}
              onClick={onLinkContainerClick}
              style={linkStyle}
            >
              <img
                className={styles.mainLogob9ffbb6svgIcon}
                loading="lazy"
                alt=""
                src={mainLogob9ffbb6svg}
                style={mainLogob9ffbb6svgIconStyle}
              />
            </div>
          </div>
          <nav className={styles.container1}>
            <div className={styles.link1}>
              <a className={styles.a}>커뮤니티</a>
            </div>
            <div className={styles.buttonmargin}>
              <div className={styles.button}>
                <a className={styles.a1} style={aStyle}>
                  대회
                </a>
              </div>
            </div>
            <div className={styles.buttonmargin1}>
              <div className={styles.button}>
                <a className={styles.a1} style={a1Style}>
                  학습
                </a>
              </div>
            </div>
            <div className={styles.margin}>
              <div className={styles.container2}>
                <div className={styles.link2}>
                  <div className={styles.label}>
                    <a className={styles.a1}>랭킹</a>
                  </div>
                </div>
              </div>
            </div>
            <div className={styles.margin1}>
              <div className={styles.container2}>
                <div className={styles.link2}>
                  <div className={styles.label}>
                    <a className={styles.a4}>더보기</a>
                  </div>
                </div>
              </div>
            </div>
          </nav>
        </div>
        <div className={styles.rectangle} style={rectangleStyle} />
        <div className={styles.container4}>
          <div className={styles.button2}>
            <img
              className={styles.imageIcon}
              loading="lazy"
              alt=""
              src="/image@2x.png"
            />
          </div>
          <div className={styles.buttonmarginWrapper}>
            <div className={styles.buttonmargin2}>
              <div className={styles.button3}>
                <a className={styles.a5}>구독 안내</a>
              </div>
            </div>
          </div>
          <div className={styles.margin2}>
            <div className={styles.link2}>
              <div className={styles.button4}>
                <div className={styles.div}>로그인</div>
              </div>
              <div className={styles.buttonmargin3}>
                <div className={styles.button5}>
                  <a className={styles.a6}>회원가입</a>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </header>
  );
};

BackgroundHorizontalBorder.propTypes = {
  className: PropTypes.string,
  mainLogob9ffbb6svg: PropTypes.string,

  /** Style props */
  linkWidth: PropTypes.any,
  aColor: PropTypes.any,
  containerColor: PropTypes.any,
  buttonWidth: PropTypes.any,
  linkBoxShadow: PropTypes.any,
  linkBackgroundColor: PropTypes.any,
  linkBorder: PropTypes.any,

  /** Action props */
  onLinkContainerClick: PropTypes.func,
};

export default BackgroundHorizontalBorder;
